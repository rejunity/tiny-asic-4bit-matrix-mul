/*
 * Copyright (c) 2024 ReJ aka Renaldas Zioma
 * SPDX-License-Identifier: Apache-2.0
 */

// refs:
// * https://dennisforbes.ca/articles/understanding-floating-point-numbers.html
// * https://github.com/oprecomp/FloatX
// * https://aclanthology.org/2023.emnlp-main.39.pdf LLM-FP4: 4-Bit Floating-Point Quantized Transformers
// * https://papers.nips.cc/paper/2020/file/13b919438259814cd5be8cb45877d577-Paper.pdf 
// * https://arxiv.org/pdf/2302.08007.pdf With Shared Microexponents, A Little Shifting Goes a Long Way
// * https://arxiv.org/pdf/2305.14314.pdf QLORA: Efficient Finetuning of Quantized LLMs
//   - table of NF4 values
// * FP4 - e3m0, e2m1
// * NF4

// `define INT4     1
// `define FP4_E3M0 1
function signed [14:0] mul_fp4_i8;
    input        [3:0] fp4;
    input signed [7:0] i8;
    begin
        mul_fp4_i8 = (|fp4[2:0] == 0) ? 0 :
                     (i8 << fp4[2:0]) * (fp4[3] ? -1: 1);
    end
endfunction

module tt_um_rejunity_fp4_mul_i8 (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // will go high when the design is enabled
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);
    assign uio_oe  = 0;         // bidirectional IOs set to INPUT
    assign uio_out = 0;         // drive bidirectional IO outputs to 0

    wire reset = ! rst_n;

    // @TODO: special signal to initiate readout
    wire       initiate_read_out = !ena;
    
    systolic_array systolic_array (
        .clk(clk),
        .reset(reset),

        .in_left({ui_in[3:0], ui_in[7:4]}),
        .in_top(uio_in),

        .restart_inputs(initiate_read_out),
        .reset_accumulators(initiate_read_out),
        .copy_accumulator_values_to_out_queue(initiate_read_out),
        .restart_out_queue(initiate_read_out),
        
        .out(uo_out)
    );

endmodule

module systolic_array #(
    parameter integer SLICES = `COMPUTE_SLICES
) (
    input  wire       clk,
    input  wire       reset,

    input  wire [7:0] in_left,
    input  wire [7:0] in_top,
    input  wire       restart_inputs,
    input  wire       reset_accumulators,
    input  wire       copy_accumulator_values_to_out_queue,
    input  wire       restart_out_queue,
    //input wire      apply_shift_to_accumulators,
    //input wire      apply_relu_to_out,

    output wire [7:0] out
);
    localparam SLICE_BITS = $clog2(SLICES);
    localparam SLICES_MINUS_1 = SLICES - 1;
    localparam W = 1 * SLICES;
    localparam H = 2 * SLICES;

    // Double buffer inputs
    // xxx_curr - arguments that are fed into MAC (multiply-accumulate) units
    // xxx_next - where the inputs are written to
    // once slice_counter reaches 0, xxx_next is flushed into xxx_curr
    reg [H*4-1:0] arg_left_curr;
    reg [W*8-1:0] arg_top_curr;

    reg [H*4-1:0] arg_left_next;
    reg [W*8-1:0] arg_top_next;

    reg  [SLICE_BITS-1:0] slice_counter;
    reg  signed [23:0] accumulators      [W*H-1:0];
    wire signed [23:0] accumulators_next [W*H-1:0];
    reg  signed [23:0] out_queue         [W*H-1:0];

    integer n;
    always @(posedge clk) begin
        if (reset | restart_inputs | slice_counter == SLICES_MINUS_1)
            slice_counter <= 0;
        else
            slice_counter <= slice_counter + 1;

        if (reset) begin
            arg_left_next <= 0;
            arg_top_next <= 0;
        end else begin // write current inputs in_xxx into the xxx_next
            arg_left_next[H*4-1 -: 8] <= in_left;
            arg_top_next [W*8-1 -: 8] <= in_top;

            if (SLICES > 1) begin // shift xxx_next for the next input
                arg_left_next   [H*4-1-8 : 0] <= arg_left_next   [H*4-1 : 8];
                arg_top_next    [W*8-1-8 : 0] <= arg_top_next    [W*8-1 : 8];
            end
        end

        if (slice_counter == 0) begin // xxx_next is flushed into xxx_curr,
                                      // once slice_counter reaches 0
            arg_left_curr <= arg_left_next;
            if (SLICES > 1)
                arg_top_curr <= {arg_top_next[7:0], arg_top_next[W*8-1: 8]};
            else
                arg_top_curr <= arg_top_next;
        end else begin // shift top systolic array arguments every clock cycle
            if (SLICES > 1)
                arg_top_curr <= {8'd0, arg_top_curr[W*8-1: 8]};
        end
        
        // The following loop must be unrolled, otherwise Verilator
        // will treat <= assignments inside the loop as errors
        // See similar bug report and workaround here:
        //   https://github.com/verilator/verilator/issues/2782
        // Ideally unroll_full Verilator metacommand should be used,
        // however it is supported only from Verilator 5.022 (#3260) [Jiaxun Yang]
        // Instead BLKLOOPINIT errors are suppressed for this loop
        /* verilator lint_off BLKLOOPINIT */
        for (n = 0; n < W*H; n = n + 1) begin
            if (reset | reset_accumulators)
                accumulators[n] <= 0;
            else
                accumulators[n] <= accumulators_next[n];

            if (copy_accumulator_values_to_out_queue) begin
                // To compensate accumulators_next 'being ahead' (shifted by 1 after computation):
                // (e.g. SLICES=4)
                // o[0] <= acc_n[3], o[1] <= acc_n[0], o[2] <= acc_n[1], o[3] <= acc_n[2]
                // o[4] <= acc_n[7], o[5] <= acc_n[4], o[6] <= acc_n[5], o[7] <= acc_n[6]
                if (n%W == 0)
                    out_queue[n] <= accumulators_next[n+W-1];
                else
                    out_queue[n] <= accumulators_next[n-1];

                // Alternatively the following code can be used
                // if additional (slice_counter-1) wait cycles are introduced
                // out_queue[n] <= accumulators_next[n];
            end else if (n > 0)
                out_queue[n-1]  <= out_queue[n];
        end
        /* verilator lint_on BLKLOOPINIT */
    end

    genvar i, j;
    generate
    for (j = 0; j < W; j = j + 1)
        for (i = 0; i < H; i = i + 1) begin : mac
            wire        [3:0] arg_0 =         arg_left_curr[i*4 +: 4];
            wire signed [7:0] arg_1 = $signed(arg_top_curr [7:0]);
            if (j == 0) begin : compute
                assign accumulators_next[i*W+W-1] =
                              accumulators[i*W+j] + mul_fp4_i8(arg_0, arg_1);
            end else begin : shift
                assign accumulators_next[i*W+j-1] =
                              accumulators[i*W+j];
            end

            // for debugging purposes in wave viewer
            wire [23:0] value_curr  = accumulators     [i*W+j];
            wire [23:0] value_next  = accumulators_next[i*W+j];
            wire [23:0] value_queue = out_queue        [i*W+j];
        end
    endgenerate

    // assign out = out_queue[0] >> 8;
    // assign out = out_queue[0][7:0];
    assign out = out_queue[0] >> (8 + 3);
endmodule
