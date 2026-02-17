// =============================================================================
// Module: systolic_array_32x32
// Description: 32x32 Output-Stationary Systolic Array for Matrix Multiplication
//              Computes C = A x B where A, B are 32x32 matrices
//              1024 PEs — optimized for RV64 AXI4 integration
//
//              Dataflow:
//              - Weights (A matrix) flow top-to-bottom (columns)
//              - Activations (B matrix) flow left-to-right (rows)
//              - Each PE accumulates partial products (output-stationary)
//
//              Architecture (abbreviated, full array is 32x32):
//                        a_col[0]  ...  a_col[31]
//                          |               |
//              b_row[0] --[PE00]-- ... --[PE0_31]-->
//                 ...
//              b_row[31]--[PE31_0]-- ... --[PE31_31]-->
//                          |               |
//
// =============================================================================

module systolic_array_32x32 #(
    parameter ARRAY_SIZE = 32,
    parameter DATA_WIDTH = 16,  // BF16
    parameter ACC_WIDTH  = 32
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         en,
    input  logic                         clear_acc,

    // Weight inputs (top edge) - one per column
    input  logic signed [DATA_WIDTH-1:0] a_col [ARRAY_SIZE],

    // Activation inputs (left edge) - one per row
    input  logic signed [DATA_WIDTH-1:0] b_row [ARRAY_SIZE],

    // Result outputs - full 32x32 result matrix
    output logic signed [ACC_WIDTH-1:0]  result [ARRAY_SIZE][ARRAY_SIZE]
);

    // Horizontal wires: b_wire[row][0] = left-edge input, [1..N] = PE pass-through
    logic signed [DATA_WIDTH-1:0] b_wire [ARRAY_SIZE][ARRAY_SIZE+1];
    // Vertical wires:   a_wire[0][col] = top-edge input,  [1..N] = PE pass-through
    logic signed [DATA_WIDTH-1:0] a_wire [ARRAY_SIZE+1][ARRAY_SIZE];

    genvar i, j;

    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : gen_a_input
            assign a_wire[0][i] = a_col[i];
        end
        for (i = 0; i < ARRAY_SIZE; i++) begin : gen_b_input
            assign b_wire[i][0] = b_row[i];
        end
    endgenerate

    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : gen_row
            for (j = 0; j < ARRAY_SIZE; j++) begin : gen_col

                processing_element #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .en         (en),
                    .clear_acc  (clear_acc),

                    // From north / from west
                    .a_in       (a_wire[i][j]),
                    .b_in       (b_wire[i][j]),

                    // To south / to east
                    .a_out      (a_wire[i+1][j]),
                    .b_out      (b_wire[i][j+1]),

                    .result_out (result[i][j])
                );

            end
        end
    endgenerate

endmodule
