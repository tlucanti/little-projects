`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 19.12.2021 17:45:15
// Design Name: 
// Module Name: SPI
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module spi8bit
(
	input				clk,
	input				reset,
	input	[7:0]		to_code_i,
	output	reg			coded_o,
	input				coded_i,
	output	reg [7:0]	decoded_o,
	output	reg			ready
);

reg		[2:0] iter;

always (posedge clk) begin
	if (reset) begin
		iter	<= 2'd0;
	end else begin

		// coder
		data_o		<= to_code_i[2'd7 - iter];

		// decoder
		decoded_o[2'd7 - iter] <= coded_i;

		if (iter == 2'd7)
			ready	<= 1'd1;
		else begin
			ready	<= 1'd0;
		end
		iter		<= iter + 2'd1;
	end
end

endmodule

module spi64bit
(
	input				clk,
	input				reset,
	input	[63:0]		to_code_i,
	output	reg			coded_o,
	input				coded_i,
	output	reg	[63:0]	decoded_o,
	output	reg			ready
);

reg		[7:0] iter;
reg		[7:0] to_code_i_8bit;
reg		[7:0] decoded_o_8bit;

wire	ready_8bit;

spi8bit		coder_decoder8bit
(
	.clk		(clk),
	.reset		(reset),
	.to_code_i	(to_code_i_8bit),
	.coded_o	(coded_o),
	.coded_i	(coded_i),
	.decoded_o	(decoded_o_8bit),
	.ready		(ready_8bit)
)

always (posedge clk) begin
	if (reset) begin
		iter	<= 7'd0;
	end else begin
		iter	<= iter + 7'd1;
		
		// coder
		if (iter == 31'd7) begin
			coded_o	<= decoded_o_8bit;
		end else if (iter == 31'd15) begin
			coded_o	<= decoded_o_8bit;
		end else if (iter == 31'd23) begin
			coded_o	<= decoded_o_8bit;
		end else if (iter == 31'd31) begin
			coded_o	<= decoded_o_8bit;
		end else if (iter == 31'd39) begin
			coded_o	<= decoded_o_8bit;
		end else if (iter == 31'd47) begin
			coded_o	<= decoded_o_8bit;
		end else if (iter == 31'd55) begin
			coded_o	<= decoded_o_8bit;
		end else if (iter == 31'd63) begin
			coded_o	<= decoded_o_8bit;

		// decoder
		if (iter == 31'd7) begin
			decoded_o[63:56]<= decoded_o_8bit;
		end else if (iter == 31'd15) begin
			decoded_o[55:48]<= decoded_o_8bit;
		end else if (iter == 31'd23) begin
			decoded_o[47:40]<= decoded_o_8bit;
		end else if (iter == 31'd31) begin
			decoded_o[39:32]<= decoded_o_8bit;
		end else if (iter == 31'd39) begin
			decoded_o[31:24]<= decoded_o_8bit;
		end else if (iter == 31'd47) begin
			decoded_o[23:16]<= decoded_o_8bit;
		end else if (iter == 31'd55) begin
			decoded_o[15:8]<= decoded_o_8bit;
		end else if (iter == 31'd63) begin
			decoded_o[7:0]<= decoded_o_8bit;
			ready <= 1'd1;
		end
	end
end
