`timescale 1ns / 1ps
////////////////////////////////////////////////////////////////////////////////
// Company: MIET
// Engineer: CHEL
// 
// Create Date: 01.04.2021 11:15:38
// Design Name: 
// Module Name: divider
// Project Name: snake game
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: LOL
// 
////////////////////////////////////////////////////////////////////////////////

`ifndef DIVIDER_H
`define DIVIDER_H

module divider(in, out);
	input	in;
	output	out;
	
	parameter speed = 5;
	reg flag;
	reg[15 : 0] cnt = speed;
	always @(posedge in) begin
		if (cnt == 0) begin
			flag <= 'b1;
			cnt <= speed;
		end
		else begin
			cnt <= cnt - 'b1;
		end
		if (flag) begin
			flag <= 'b0;
		end
	end
	assign out = flag;
endmodule

`endif
