
module W25Q16 (
	input		CS,		// 1 // chip select input
	output reg	DO,		// 2 // data output
	input		WP,		// 3 // write protect input
						// 4 // GND
	input		DI,		// 5 // data input
	input		CLK,	// 6 // serial clock input
	input		HOLD	// 7 // hold input
						// 8 // VCCd
);

always @(posedge CLK) begin
	if (CS) begin
		// Device is deselected
		DO <= 1'b1;
		
	end
end

endmodule
