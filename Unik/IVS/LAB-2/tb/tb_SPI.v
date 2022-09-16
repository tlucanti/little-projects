`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05.12.2021 17:45:15
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


module tb_SPI();

reg		clk;
reg		reset;
reg		miso = 0;

wire	mosi;
wire	flash;
wire	sensor;

reg	[31:0] counter = 0;
reg	[63:0] flash_data = 64'b1010101010101010101010101010101010101010101010101010101010101010;
reg [31:0] i1 = 0;
reg [31:0] i2 = 0;
reg [31:0] i3 = 0;
reg [31:0] i4 = 0;

master dut
(
	.CLK		(clk),
	.RESET		(reset),
	.MISO		(miso),
	.MOSI		(mosi),
	.FLASH_CS	(flash),
	.SENSOR_CS	(sensor)
);



always #5 clk = ~clk;
initial begin
	clk = 0;
	reset = 0;
	@(negedge clk);
	reset = 1;
	repeat(2) begin
		@(negedge clk);
	end
	reset = 0;

	// instr read
	repeat(8) begin
		#1000
		counter <= 1;
		miso <= 0;
	end

	// read addr
	repeat(24) begin
		#1000
		counter <= 2;
	end

	// read data
	repeat(64) begin
		#1000
		counter <= 3;
		miso <= flash_data[i1];
		i1 <= i1 + 1;
	end

	// led 1
	repeat(64) begin
		#1000
		counter <= 4;
		miso <= 0;
	end

	// idle
	#1000
	counter <= 5;

	// write enable
	repeat(8) begin
		#1000
		counter <= 6;
		miso <= 0;
	end

	// idle
	#1000
	counter <= 7;

	// program instr
	repeat(8) begin
		#1000
		counter <= 8;
		miso <= 0;
	end

	// write addr
	repeat(24) begin
		#1000
		counter <= 9;
		miso <= 0;
	end

	// program flash
	repeat(64) begin
		#1000
		counter <= 10;
		miso <= 0;
	end

	// idle
	#1000
	counter <= 11;

	// write disable
	repeat(8) begin
		#1000
		counter <= 12;
		miso <= 0;
	end

	// idle
	#1000
	counter <= 13;

	// read fast instr
	repeat(8) begin
		#1000
		counter <= 14;
		miso <= 0;
	end

	// read fast addr
	repeat(24) begin
		#1000
		counter <= 15;
		miso <= 0;
	end

	// read fast data
	repeat(64) begin
		#1000
		counter <= 16;
		miso <= flash_data[i2];
		i2 <= i2 + 1;
	end

	// led 2
	repeat(64) begin
		#1000
		counter <= 17;
		miso <= 0;
	end

	// instr read
	repeat(8) begin
		#1000
		counter <= 18;
		miso <= 0;
	end

	// sensor read addr
	repeat(24) begin
		#1000
		counter <= 19;
	end

	// sensor read data
	repeat(64) begin
		#1000
		counter <= 20;
		miso <= flash_data[i3];
		i3 <= i3 + 1;
	end

	// led 3
	repeat(64) begin
		#1000
		counter <= 21;
		miso <= 0;
	end

	counter <= 22;
	#2000000;
	
	$finish();
end

endmodule
