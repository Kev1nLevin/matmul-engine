// =============================================================================
// Module: tpu_top
// Description: Integration top-level — connects the AXI4 slave wrapper
//              to the 32x32 systolic array TPU.
//
//              Hierarchy:
//                tpu_top
//                ├── tpu_axi4_wrapper   (AXI4 slave, MMIO, IRQ)
//                └── systolic_top       (32x32 systolic array)
//                      ├── input_skew_controller (×2)
//                      ├── systolic_array_32x32
//                      │     └── processing_element ×1024
//                      │           └── mac_unit (BF16→FP32, 2-stage)
//                      ├── systolic_controller
//                      └── gelu_unit ×32 (result read path)
//
// Connect to a RV64 AXI4 master (e.g., CVA6, Rocket, BOOM).
// Map this block to an address in the CPU's memory space; the irq output
// connects to the CPU's external interrupt pin (MEIP in mip/mie).
// =============================================================================

module tpu_top #(
    parameter ARRAY_SIZE     = 32,
    parameter DATA_WIDTH     = 16,  // BF16
    parameter ACC_WIDTH      = 32,
    parameter AXI_ID_WIDTH   = 4,
    parameter AXI_ADDR_WIDTH = 64,   // RV64
    parameter AXI_DATA_WIDTH = 64,   // 64-bit bus
    // Derived (must match systolic_top)
    parameter ADDR_WIDTH     = $clog2(ARRAY_SIZE),
    parameter AXI_STRB_WIDTH = AXI_DATA_WIDTH / 8
)(
    input  logic clk,
    input  logic rst_n,

    // =========================================================================
    // AXI4 Slave — connect to RV32 CPU AXI master port
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
    output logic [1:0]                 s_axi_bresp,
    output logic                       s_axi_bvalid,
    input  logic                       s_axi_bready,

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
    output logic [1:0]                 s_axi_rresp,
    output logic                       s_axi_rlast,
    output logic                       s_axi_rvalid,
    input  logic                       s_axi_rready,

    // =========================================================================
    // Interrupt — connect to CPU MEIP (machine external interrupt pin)
    // =========================================================================
    output logic irq
);

    // =========================================================================
    // Internal wires between wrapper and systolic_top
    // =========================================================================
    logic                         tpu_wr_en_a;
    logic                         tpu_wr_en_b;
    logic [ADDR_WIDTH-1:0]        tpu_wr_row_addr;
    logic signed [DATA_WIDTH-1:0] tpu_wr_data [ARRAY_SIZE];

    logic                         tpu_start;
    logic                         tpu_busy;
    logic                         tpu_done;

    logic [ADDR_WIDTH-1:0]        tpu_rd_row_addr;
    logic signed [ACC_WIDTH-1:0]  tpu_rd_data [ARRAY_SIZE];

    logic                         tpu_gelu_en;

    // =========================================================================
    // AXI4 Wrapper
    // =========================================================================
    tpu_axi4_wrapper #(
        .ARRAY_SIZE     (ARRAY_SIZE),
        .DATA_WIDTH     (DATA_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .AXI_ID_WIDTH   (AXI_ID_WIDTH),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH)
    ) u_axi4_wrapper (
        .clk            (clk),
        .rst_n          (rst_n),

        // AXI4 write address
        .s_axi_awid     (s_axi_awid),
        .s_axi_awaddr   (s_axi_awaddr),
        .s_axi_awlen    (s_axi_awlen),
        .s_axi_awsize   (s_axi_awsize),
        .s_axi_awburst  (s_axi_awburst),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),

        // AXI4 write data
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wlast    (s_axi_wlast),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),

        // AXI4 write response
        .s_axi_bid      (s_axi_bid),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),

        // AXI4 read address
        .s_axi_arid     (s_axi_arid),
        .s_axi_araddr   (s_axi_araddr),
        .s_axi_arlen    (s_axi_arlen),
        .s_axi_arsize   (s_axi_arsize),
        .s_axi_arburst  (s_axi_arburst),
        .s_axi_arvalid  (s_axi_arvalid),
        .s_axi_arready  (s_axi_arready),

        // AXI4 read data
        .s_axi_rid      (s_axi_rid),
        .s_axi_rdata    (s_axi_rdata),
        .s_axi_rresp    (s_axi_rresp),
        .s_axi_rlast    (s_axi_rlast),
        .s_axi_rvalid   (s_axi_rvalid),
        .s_axi_rready   (s_axi_rready),

        // TPU interface
        .tpu_wr_en_a    (tpu_wr_en_a),
        .tpu_wr_en_b    (tpu_wr_en_b),
        .tpu_wr_row_addr(tpu_wr_row_addr),
        .tpu_wr_data    (tpu_wr_data),
        .tpu_start      (tpu_start),
        .tpu_busy       (tpu_busy),
        .tpu_done       (tpu_done),
        .tpu_rd_row_addr(tpu_rd_row_addr),
        .tpu_rd_data    (tpu_rd_data),
        .gelu_en        (tpu_gelu_en),
        .irq            (irq)
    );

    // =========================================================================
    // Systolic Array TPU
    // =========================================================================
    systolic_top #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_systolic_top (
        .clk         (clk),
        .rst_n       (rst_n),

        .wr_en_a     (tpu_wr_en_a),
        .wr_en_b     (tpu_wr_en_b),
        .wr_row_addr (tpu_wr_row_addr),
        .wr_data     (tpu_wr_data),

        .start       (tpu_start),
        .busy        (tpu_busy),
        .done        (tpu_done),

        .rd_row_addr (tpu_rd_row_addr),
        .rd_data     (tpu_rd_data),
        .gelu_en     (tpu_gelu_en)
    );

endmodule
