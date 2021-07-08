`timescale 1ns / 1ps
////////////////////////////////////////////////////////////////////////////////
// Company: MIET
// Engineer: CHEL
// 
// Create Date: 01.04.2021 11:15:38
// Design Name: 
// Module Name: random
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

`ifndef RANDOM_H
`define RANDOM_H

module random(clk, num);
	input				clk;
	output	[n - 1:0]	num;

	parameter	min		= 1;	// random maximum value
	parameter	max		= 256;	// random minimum value
	parameter	n		= 32;	// output bus size
	parameter	seed	= 0;

	reg		[31:0]	cnt = 0;
	wire		[31:0]	maxrm=max;
	wire		[31:0]	minrm=min;
	wire		[n-1:0]	maxr=maxrm[n-1:0];
	wire		[n-1:0]	minr=minrm[n-1:0];
	wire	[31:0]	sum = 0;

	always @(posedge clk) begin
		cnt <= (cnt + 1 + seed);// ^ (seed + 1);
	end

	// assign	sum = cnt % (max - min) + min;
	assign	num = cnt[n-1:0] % (maxr - minr) + minr - 1;
endmodule

`endif
