// =============================================================================
// Module: processing_element
// Description: Systolic Array Processing Element (PE)
//              Wraps a MAC unit and adds systolic data flow registers.
//              - Weight (a) flows top-to-bottom (vertical)
//              - Activation (b) flows left-to-right (horizontal)
//              This implements the classic output-stationary dataflow.
// =============================================================================

module processing_element #(
    parameter DATA_WIDTH = 16,  // BF16
    parameter ACC_WIDTH  = 32
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         en,
    input  logic                         clear_acc,

    // Systolic data inputs (from north / from west)
    input  logic signed [DATA_WIDTH-1:0] a_in,       // Weight from north neighbor
    input  logic signed [DATA_WIDTH-1:0] b_in,       // Activation from west neighbor

    // Systolic data outputs (to south / to east)
    output logic signed [DATA_WIDTH-1:0] a_out,      // Weight to south neighbor
    output logic signed [DATA_WIDTH-1:0] b_out,      // Activation to east neighbor

    // Result output
    output logic signed [ACC_WIDTH-1:0]  result_out  // Accumulated result
);

    // Systolic pass-through: 1-cycle delay propagates data to neighbor PEs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
        end
    end


    mac_unit #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_mac (
        .clk       (clk),
        .rst_n     (rst_n),
        .en        (en),
        .clear_acc (clear_acc),
        .a_in      (a_in),
        .b_in      (b_in),
        .acc_out   (result_out)
    );

endmodule
