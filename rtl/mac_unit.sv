// =============================================================================
// Module: mac_unit
// Description: 2-Stage Pipelined BF16 Multiply-Accumulate Unit
//              Computes: acc = acc + (a * b)
//              Inputs:      BF16  (16-bit brain-float)
//              Accumulator: FP32  (32-bit single-precision)
//
//              Pipeline:
//                Stage 1 (registered): BF16 × BF16 → FP32 product
//                Stage 2 (registered): FP32 acc   += FP32 product
//
//              Simplified Option-A floating point:
//                - Flush-to-zero for denormals (exponent == 0)
//                - No NaN / Inf propagation
//                - Truncate rounding (round toward zero)
//                - Normal numbers only on both input and accumulator paths
//
//              BF16 format: [15]=sign [14:7]=exp(bias=127) [6:0]=mantissa
//              FP32 format: [31]=sign [30:23]=exp(bias=127) [22:0]=mantissa
//
// MAC Latency: 2 clock cycles (MAC_LATENCY localparam — consumed by controller)
// =============================================================================

module mac_unit #(
    parameter DATA_WIDTH = 16,   // BF16 input operands
    parameter ACC_WIDTH  = 32    // FP32 accumulator
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         en,
    input  logic                         clear_acc,
    input  logic [DATA_WIDTH-1:0]        a_in,    // BF16 weight
    input  logic [DATA_WIDTH-1:0]        b_in,    // BF16 activation
    output logic [ACC_WIDTH-1:0]         acc_out  // FP32 result
);

    localparam MAC_LATENCY = 2;  // stages — systolic_controller must add (MAC_LATENCY-1) drain cycles

    // =========================================================================
    // Stage 1: BF16 × BF16 → FP32 product
    // =========================================================================

    // Extract BF16 fields
    logic        a_s;
    logic [7:0]  a_e;
    logic [7:0]  a_mf;   // full mantissa with implicit 1: {1, a_in[6:0]}
    logic        b_s;
    logic [7:0]  b_e;
    logic [7:0]  b_mf;

    assign a_s  = a_in[15];
    assign a_e  = a_in[14:7];
    assign a_mf = {1'b1, a_in[6:0]};
    assign b_s  = b_in[15];
    assign b_e  = b_in[14:7];
    assign b_mf = {1'b1, b_in[6:0]};

    // Mantissa multiply: 8b × 8b = 16b
    logic [15:0] mant_prod;
    assign mant_prod = a_mf * b_mf;

    // Product exponent: sum of input exponents minus one bias
    logic [8:0] prod_exp_raw;   // 9-bit to detect underflow (bit 8 = borrow/sign)
    assign prod_exp_raw = {1'b0, a_e} + {1'b0, b_e} - 9'd127;

    // Combinational FP32 product
    logic [ACC_WIDTH-1:0] product_comb;

    always_comb begin
        if (a_e == 8'h00 || b_e == 8'h00) begin
            // Zero or denormal input → flush product to zero
            product_comb = '0;
        end else if (prod_exp_raw[8]) begin
            // Exponent underflowed (a_e + b_e < 127) → flush to zero
            product_comb = '0;
        end else if (mant_prod[15]) begin
            // Product in [2.0, 4.0): exp+1, mantissa from mant_prod[14:0]
            product_comb = {a_s ^ b_s, prod_exp_raw[7:0] + 8'd1,
                            mant_prod[14:0], 8'b0};
        end else begin
            // Product in [1.0, 2.0): mantissa from mant_prod[13:0]
            product_comb = {a_s ^ b_s, prod_exp_raw[7:0],
                            mant_prod[13:0], 9'b0};
        end
    end

    // Stage 1 register
    logic [ACC_WIDTH-1:0] product_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            product_reg <= '0;
        else if (clear_acc)
            product_reg <= '0;
        else if (en)
            product_reg <= product_comb;
    end

    // =========================================================================
    // Stage 2: FP32 Accumulate  (acc_reg += product_reg)
    // =========================================================================

    // Extract FP32 fields from accumulator and product
    logic        x_s, y_s;     // signs
    logic [7:0]  x_e, y_e;     // exponents
    logic [23:0] x_m, y_m;     // mantissas with implicit leading 1 (24 bits)

    logic [ACC_WIDTH-1:0] acc_reg;

    assign x_s = acc_reg[31];
    assign x_e = acc_reg[30:23];
    assign x_m = (x_e == 8'h00) ? 24'b0 : {1'b1, acc_reg[22:0]};   // flush denormal

    assign y_s = product_reg[31];
    assign y_e = product_reg[30:23];
    assign y_m = (y_e == 8'h00) ? 24'b0 : {1'b1, product_reg[22:0]};

    // ---- Sort by magnitude so big_e >= sml_e ----
    logic        big_is_y;
    assign big_is_y = (y_e > x_e) || ((y_e == x_e) && (y_m > x_m));

    logic        big_s, sml_s;
    logic [7:0]  big_e, sml_e;
    logic [23:0] big_m, sml_m;

    assign {big_s, big_e, big_m} = big_is_y ? {y_s, y_e, y_m} : {x_s, x_e, x_m};
    assign {sml_s,        sml_m} = big_is_y ? {x_s,       x_m} : {y_s,       y_m};
    assign sml_e                 = big_is_y ? x_e              : y_e;

    // ---- Align smaller operand ----
    // exp_diff = big_e - sml_e, always ≥ 0
    logic [7:0]  exp_diff;
    logic [4:0]  align_shift;
    logic [23:0] sml_m_aligned;

    assign exp_diff    = big_e - sml_e;
    assign align_shift = (exp_diff >= 8'd24) ? 5'd24 : exp_diff[4:0];
    assign sml_m_aligned = sml_m >> align_shift;  // right shift; bits fall off = truncate

    // ---- Effective operation: add or subtract mantissas ----
    logic add_op;
    assign add_op = (big_s == sml_s);   // same sign → add magnitudes

    // Addition path (25-bit to capture possible carry)
    logic [24:0] add_result;
    assign add_result = {1'b0, big_m} + {1'b0, sml_m_aligned};

    // Subtraction path (big_m ≥ sml_m_aligned guaranteed)
    logic [23:0] sub_result;
    assign sub_result = big_m - sml_m_aligned;

    // Leading-zero count for normalization: highest set bit k → lz_count = 23-k
    logic [4:0] lz_count;
    always_comb begin
        lz_count = 5'd24;   // default: all zeros
        for (int k = 0; k <= 23; k++)
            if (sub_result[k]) lz_count = 5'(23 - k);
    end

    logic [23:0] sub_norm;
    assign sub_norm = sub_result << lz_count;  // sub_norm[23] = implicit 1 after normalization

    // ---- Build accumulator next value ----
    logic [ACC_WIDTH-1:0] acc_next;

    always_comb begin
        if (x_e == 8'h00 && y_e == 8'h00) begin
            // Both operands are zero
            acc_next = '0;

        end else if (x_e == 8'h00) begin
            // Accumulator is zero — result is the product
            acc_next = product_reg;

        end else if (y_e == 8'h00) begin
            // Product is zero — accumulator unchanged
            acc_next = acc_reg;

        end else if (add_op) begin
            // ---- Addition path ----
            if (add_result[24]) begin
                // Carry out: result = 1xx.xxx → shift right 1, exp+1
                acc_next = {big_s, big_e + 8'd1, add_result[23:1]};
            end else begin
                // No carry: result already normalized (implicit 1 at bit 23)
                acc_next = {big_s, big_e, add_result[22:0]};
            end

        end else begin
            // ---- Subtraction path ----
            if (sub_result == '0) begin
                // Exact cancellation
                acc_next = '0;
            end else if (big_e >= {3'b0, lz_count}) begin
                // Shift left, decrement exponent — stays in normal range
                acc_next = {big_s, big_e - {3'b0, lz_count}, sub_norm[22:0]};
            end else begin
                // Exponent underflows — flush to zero
                acc_next = '0;
            end
        end
    end

    // Stage 2 register (accumulator)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= '0;
        else if (clear_acc)
            acc_reg <= '0;
        else if (en)
            acc_reg <= acc_next;
    end

    assign acc_out = acc_reg;

endmodule
