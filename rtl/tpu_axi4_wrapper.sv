// =============================================================================
// Module: tpu_axi4_wrapper
// Description: Full AXI4 slave wrapper for the 32x32 systolic array TPU.
//              Bridges a RV64 AXI4 master (e.g., CVA6, Rocket, BOOM) to the
//              systolic_top handshake interface.
//
// Memory Map (base-relative, 64-bit AXI data bus):
//
//   0x0000_0000_0000_0000 : CTRL
//                           [bit4=gelu_en, bit3=irq_en, bit2=done(RO), bit1=busy(RO), bit0=start(WO)]
//
//   0x0000_0000_0000_1000 : MAT_A
//                           Row i at offset (i * MAT_ROW_BYTES).
//                           INCR burst of WORDS_PER_ROW beats (AWLEN = WORDS_PER_ROW-1).
//                           Each 64-bit word carries 4 BF16 values (little-endian).
//                           MAT_ROW_BYTES = ARRAY_SIZE * DATA_WIDTH / 8 = 64 bytes
//                           WORDS_PER_ROW = 32 * 16 / 64 = 8  →  AWLEN = 7
//
//   0x0000_0000_0000_2000 : MAT_B   (same layout as MAT_A)
//
//   0x0000_0000_0000_3000 : RESULTS
//                           Row i at offset (i * RESULT_ROW_BYTES).
//                           INCR burst of RESULT_WORDS_PER_ROW beats (ARLEN = 15).
//                           Each 64-bit word carries 2 x FP32 accumulators.
//                           RESULT_ROW_BYTES = ARRAY_SIZE * ACC_WIDTH / 8 = 128 bytes
//                           RESULT_WORDS_PER_ROW = 32 / 2 = 16  →  ARLEN = 15
//
// Write Protocol:
//   - Matrix rows : INCR burst, AWLEN = WORDS_PER_ROW-1   (8 beats for 32x32 BF16)
//   - CTRL write  : single beat, AWLEN = 0
//
// Read Protocol:
//   - Result rows : INCR burst, ARLEN = RESULT_WORDS_PER_ROW-1  (16 beats for 32x32)
//   - CTRL read   : single beat, ARLEN = 0
//
// Notes:
//   - Only INCR burst type (AWBURST/ARBURST = 2'b01) is supported.
//   - BRESP and RRESP are always OKAY (2'b00).
//   - done is latched; cleared when a new start is issued.
//   - irq is level-sensitive: irq = irq_en & done_latched.
//   - Write and read channels are fully independent (separate FSMs).
// =============================================================================

module tpu_axi4_wrapper #(
    parameter ARRAY_SIZE     = 32,
    parameter DATA_WIDTH     = 16,  // BF16
    parameter ACC_WIDTH      = 32,
    parameter AXI_ID_WIDTH   = 4,
    parameter AXI_ADDR_WIDTH = 64,   // RV64 address space
    parameter AXI_DATA_WIDTH = 64,   // 64-bit data bus
    // Derived — do not override
    parameter ADDR_WIDTH         = $clog2(ARRAY_SIZE),
    parameter AXI_STRB_WIDTH     = AXI_DATA_WIDTH / 8,
    parameter WORDS_PER_ROW      = (ARRAY_SIZE * DATA_WIDTH) / AXI_DATA_WIDTH,
    parameter RESULTS_PER_WORD   = AXI_DATA_WIDTH / ACC_WIDTH,
    parameter RESULT_WORDS_PER_ROW = ARRAY_SIZE / RESULTS_PER_WORD,
    // Row stride log2 for address decode
    parameter MAT_ROW_STRIDE_LOG2    = $clog2(ARRAY_SIZE * DATA_WIDTH / 8),
    parameter RESULT_ROW_STRIDE_LOG2 = $clog2(ARRAY_SIZE * ACC_WIDTH  / 8)
)(
    input  logic clk,
    input  logic rst_n,

    // =========================================================================
    // AXI4 Slave Interface
    // =========================================================================

    // Write Address Channel
    input  logic [AXI_ID_WIDTH-1:0]   s_axi_awid,
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic [7:0]                s_axi_awlen,
    input  logic [2:0]                s_axi_awsize,
    input  logic [1:0]                s_axi_awburst,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,

    // Write Data Channel
    input  logic [AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  logic [AXI_STRB_WIDTH-1:0] s_axi_wstrb,
    input  logic                      s_axi_wlast,
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,

    // Write Response Channel
    output logic [AXI_ID_WIDTH-1:0]   s_axi_bid,
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,

    // Read Address Channel
    input  logic [AXI_ID_WIDTH-1:0]   s_axi_arid,
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic [7:0]                s_axi_arlen,
    input  logic [2:0]                s_axi_arsize,
    input  logic [1:0]                s_axi_arburst,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,

    // Read Data Channel
    output logic [AXI_ID_WIDTH-1:0]   s_axi_rid,
    output logic [AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rlast,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,

    // =========================================================================
    // systolic_top Interface
    // =========================================================================
    output logic                         tpu_wr_en_a,
    output logic                         tpu_wr_en_b,
    output logic [ADDR_WIDTH-1:0]        tpu_wr_row_addr,
    output logic signed [DATA_WIDTH-1:0] tpu_wr_data [ARRAY_SIZE],

    output logic                         tpu_start,
    input  logic                         tpu_busy,
    input  logic                         tpu_done,

    output logic [ADDR_WIDTH-1:0]        tpu_rd_row_addr,
    input  logic signed [ACC_WIDTH-1:0]  tpu_rd_data [ARRAY_SIZE],

    output logic                         gelu_en,
    output logic                         irq
);

    // =========================================================================
    // Address region constants (bits [15:12] of address)
    // =========================================================================
    localparam [3:0] REGION_CTRL    = 4'h0;
    localparam [3:0] REGION_MAT_A   = 4'h1;
    localparam [3:0] REGION_MAT_B   = 4'h2;
    localparam [3:0] REGION_RESULTS = 4'h3;

    // Bit-widths for beat counters
    localparam WR_BEAT_BITS = $clog2(WORDS_PER_ROW);          // 3 for 32x32 BF16 @64b
    localparam RD_BEAT_BITS = $clog2(RESULT_WORDS_PER_ROW);   // 4 for 32x32 FP32 @64b

    // =========================================================================
    // CTRL register
    // =========================================================================
    logic ctrl_irq_en;
    logic ctrl_gelu_en;
    logic ctrl_done;    // latched; cleared on start

    // =========================================================================
    // Write FSM
    // =========================================================================
    typedef enum logic [1:0] {
        WR_IDLE   = 2'b00,
        WR_DATA   = 2'b01,
        WR_COMMIT = 2'b10,
        WR_RESP   = 2'b11
    } wr_state_t;

    wr_state_t wr_state;

    logic [AXI_ID_WIDTH-1:0]   aw_id_reg;
    logic [AXI_ADDR_WIDTH-1:0] aw_addr_reg;

    // Combinational address decode (from registered AW address)
    logic [3:0]            wr_region;
    logic [ADDR_WIDTH-1:0] wr_row;

    assign wr_region = aw_addr_reg[15:12];
    assign wr_row    = aw_addr_reg[MAT_ROW_STRIDE_LOG2 + ADDR_WIDTH - 1 -: ADDR_WIDTH];
    //   For 32x32: MAT_ROW_STRIDE_LOG2=5, ADDR_WIDTH=5 → bits [9:5]

    wire wr_is_ctrl  = (wr_region == REGION_CTRL);
    wire wr_is_mat_a = (wr_region == REGION_MAT_A);
    wire wr_is_mat_b = (wr_region == REGION_MAT_B);

    // Row write buffer: accumulates WORDS_PER_ROW 64-bit words
    logic [AXI_DATA_WIDTH-1:0]  row_buf [WORDS_PER_ROW];
    logic [WR_BEAT_BITS-1:0]    wr_beat_cnt;

    logic tpu_start_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state    <= WR_IDLE;
            aw_id_reg   <= '0;
            aw_addr_reg <= '0;
            wr_beat_cnt <= '0;
            for (int i = 0; i < WORDS_PER_ROW; i++) row_buf[i] <= '0;
            ctrl_irq_en  <= 1'b0;
            ctrl_gelu_en <= 1'b0;
            ctrl_done    <= 1'b0;
            tpu_start_r <= 1'b0;
        end else begin
            tpu_start_r <= 1'b0;

            if (tpu_done)
                ctrl_done <= 1'b1;

            case (wr_state)

                WR_IDLE: begin
                    if (s_axi_awvalid) begin
                        aw_id_reg   <= s_axi_awid;
                        aw_addr_reg <= s_axi_awaddr;
                        wr_beat_cnt <= '0;
                        wr_state    <= WR_DATA;
                    end
                end

                WR_DATA: begin
                    if (s_axi_wvalid) begin
                        if (wr_is_mat_a || wr_is_mat_b)
                            row_buf[wr_beat_cnt] <= s_axi_wdata;

                        if (wr_is_ctrl) begin
                            ctrl_irq_en  <= s_axi_wdata[3];
                            ctrl_gelu_en <= s_axi_wdata[4];
                            if (s_axi_wdata[0]) begin
                                tpu_start_r <= 1'b1;
                                ctrl_done   <= 1'b0;
                            end
                        end

                        if (s_axi_wlast) begin
                            wr_beat_cnt <= '0;
                            wr_state <= (wr_is_mat_a || wr_is_mat_b) ? WR_COMMIT : WR_RESP;
                        end else begin
                            wr_beat_cnt <= wr_beat_cnt + 1;
                        end
                    end
                end

                WR_COMMIT: begin
                    // tpu_wr_en_a / tpu_wr_en_b pulse for exactly one cycle here
                    wr_state <= WR_RESP;
                end

                WR_RESP: begin
                    if (s_axi_bready)
                        wr_state <= WR_IDLE;
                end

            endcase
        end
    end

    assign s_axi_awready = (wr_state == WR_IDLE);
    assign s_axi_wready  = (wr_state == WR_DATA);
    assign s_axi_bvalid  = (wr_state == WR_RESP);
    assign s_axi_bid     = aw_id_reg;
    assign s_axi_bresp   = 2'b00;

    assign tpu_wr_en_a     = (wr_state == WR_COMMIT) && wr_is_mat_a;
    assign tpu_wr_en_b     = (wr_state == WR_COMMIT) && wr_is_mat_b;
    assign tpu_wr_row_addr = wr_row;
    assign tpu_start       = tpu_start_r;

    // Unpack row_buf into ARRAY_SIZE BF16 values
    // Each 64-bit word holds (AXI_DATA_WIDTH/DATA_WIDTH) = 4 BF16 values
    generate
        for (genvar w = 0; w < WORDS_PER_ROW; w++) begin : gen_unpack_word
            for (genvar b = 0; b < AXI_DATA_WIDTH / DATA_WIDTH; b++) begin : gen_unpack_byte
                assign tpu_wr_data[w * (AXI_DATA_WIDTH / DATA_WIDTH) + b] =
                    row_buf[w][b * DATA_WIDTH +: DATA_WIDTH];
            end
        end
    endgenerate

    // =========================================================================
    // Read FSM
    // =========================================================================
    typedef enum logic {
        RD_IDLE = 1'b0,
        RD_DATA = 1'b1
    } rd_state_t;

    rd_state_t rd_state;

    logic [AXI_ID_WIDTH-1:0]   ar_id_reg;
    logic [AXI_ADDR_WIDTH-1:0] ar_addr_reg;
    logic [7:0]                ar_len_reg;
    logic [7:0]                rd_beat_cnt;

    // Combinational decode (from registered AR address)
    logic [3:0]            rd_region;
    logic [ADDR_WIDTH-1:0] rd_row;

    assign rd_region = ar_addr_reg[15:12];
    assign rd_row    = ar_addr_reg[RESULT_ROW_STRIDE_LOG2 + ADDR_WIDTH - 1 -: ADDR_WIDTH];
    //   For 32x32: RESULT_ROW_STRIDE_LOG2=7, ADDR_WIDTH=5 → bits [11:7]

    wire rd_is_ctrl = (rd_region == REGION_CTRL);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state    <= RD_IDLE;
            ar_id_reg   <= '0;
            ar_addr_reg <= '0;
            ar_len_reg  <= '0;
            rd_beat_cnt <= '0;
        end else begin
            case (rd_state)

                RD_IDLE: begin
                    if (s_axi_arvalid) begin
                        ar_id_reg   <= s_axi_arid;
                        ar_addr_reg <= s_axi_araddr;
                        ar_len_reg  <= s_axi_arlen;
                        rd_beat_cnt <= '0;
                        rd_state    <= RD_DATA;
                    end
                end

                RD_DATA: begin
                    if (s_axi_rready) begin
                        if (rd_beat_cnt == ar_len_reg)
                            rd_state <= RD_IDLE;
                        else
                            rd_beat_cnt <= rd_beat_cnt + 8'd1;
                    end
                end

            endcase
        end
    end

    assign s_axi_arready = (rd_state == RD_IDLE);
    assign s_axi_rvalid  = (rd_state == RD_DATA);
    assign s_axi_rlast   = (rd_state == RD_DATA) && (rd_beat_cnt == ar_len_reg);
    assign s_axi_rid     = ar_id_reg;
    assign s_axi_rresp   = 2'b00;

    assign tpu_rd_row_addr = rd_row;

    // -------------------------------------------------------------------------
    // Read data mux
    // CTRL  : single word with status bits
    // RESULT: pack RESULTS_PER_WORD accumulators per 64-bit beat
    //         beat k  →  { C[i][k*2+1], C[i][k*2] }   (for RESULTS_PER_WORD=2)
    // -------------------------------------------------------------------------
    always_comb begin
        s_axi_rdata = '0;
        if (rd_is_ctrl) begin
            s_axi_rdata = AXI_DATA_WIDTH'({ctrl_gelu_en, ctrl_irq_en, ctrl_done, tpu_busy, 1'b0});
        end else begin
            for (int r = 0; r < RESULTS_PER_WORD; r++) begin
                automatic int col = int'(rd_beat_cnt) * RESULTS_PER_WORD + r;
                s_axi_rdata[r * ACC_WIDTH +: ACC_WIDTH] =
                    ACC_WIDTH'(signed'(tpu_rd_data[col]));
            end
        end
    end

    // =========================================================================
    // IRQ and GeLU enable
    // =========================================================================
    assign irq     = ctrl_irq_en & ctrl_done;
    assign gelu_en = ctrl_gelu_en;

endmodule
