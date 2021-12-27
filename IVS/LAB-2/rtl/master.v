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

module master #
(
	parameter	READ_ADDR		= 123,
	parameter	READ_ADDR_FAST	= 456,
	parameter	WRITE_ADDR		= 789,
	parameter	READ_ADDR_MPU	= 114
)
(
	input	CLK,				// serial clock
	input	RESET,				// reset signal
	input	MISO,				// master input slave output
	output	reg MOSI,			// master output slave input	
	output	reg FLASH_CS,		// flash memory chip select
	output	reg SENSOR_CS		// sendor chip select
);

// spi		coder_decoder
// (
// 	.clk		(CLK),
// 	.to_code_i	(coder_data),
// 	.coded_o	(MOSI),
// 	.coded_i	(MISO),
// 	.decoded_o	(decoder_data),
// 	.ready		(ready)
// );

// ----------------------------- LOCAL PARAMETERS ------------------------------
localparam	READ_DATA_INSTR		= 8'h3;		// flash read data instruction
localparam	WRITE_ENABLE_INSTR	= 8'h6;		// flash write enable instruction
localparam	WRITE_DATA_INSTR	= 8'h2;		// flash program data instruction
localparam	WRITE_DISABLE_INSTR	= 8'h4;		// flash write disable instruction
localparam	FAST_READ_INSTR		= 8'h0;		// flash fast read instruction

// ------------------------------ MASTER STATES  -------------------------------
localparam	IDLE				= 8'h00;		
/*
 * default master state
 */
localparam	FLASH_ISNTR_READ	= 8'h01;
/*
 * flash first read data instruction (read) sending state
 */
localparam	FLASH_ADDR_READ		= 8'h02;
/*
 * flash first read data address sending state
 */
localparam	FLASH_READ_STATE	= 8'h03;
/*
 * flash first data receving
 */
localparam	LED_GLOW_1			= 8'h04;
/*
 * first show red data from flash in LED state (from flash read)
 */
localparam	FLASH_WRITE_ENABLE	= 8'h05;
/*
 * flash write enable flash state
 */
localparam	FLASH_INSTR_WRITE	= 8'h06;
/*
 * flash write data instruction (write) sending state
 */
localparam	FLASH_ADDR_WRITE	= 8'h07;
/*
 * flash write data address sending state
 */
localparam	FLASH_WRITE_STATE	= 8'h08;
/*
 * flash program data state
 */
localparam	FLASH_WRITE_DISABLE	= 8'h09;
/*
 * flash wrote disable state
 */
localparam	FLASH_INSTR_FAST	= 8'h0a;
/*
 * flash second read instruction (fast read) sending state
 */
localparam	FLASH_ADDR_FAST		= 8'h0b;
/*
 * flash second read data address sending state
 */
localparam	FLASH_FAST_DUMMY	= 8'h0c;
/*
 * flash dummy byte state for fast read command
 */
localparam	FLASH_FAST_READ		= 8'h0d;
/*
 * flash second data receving
 */
localparam	LED_GLOW_2			= 8'h0e;
/*
 * second show red data from flash in LED state (from flash fast read)
 */
localparam	SENSOR_ISNTR_READ	= 8'h0f;
/*
 * sendor first read data instruction (read) sending state
 */
localparam	SENSOR_ADDR_READ	= 8'h10;
/*
 * sensor first read data address sending state
 */
localparam	SENSOR_READ_STATE	= 8'h11;
/*
 * sensor first data receving
 */
localparam	LED_GLOW_3			= 8'h12;
/*
 * third show red data from flash in LED state (from sensor)
 */
localparam	TERMINATE			= 8'h13;
/*
 * final infinity terminate state
 */

// --------------------------------- REGISTERS ---------------------------------
reg		[23:0]	flash_addr;			// addres for flash
reg		[7:0]	flash_instr;		// instruction for flash
reg		[63:0]	flash_data;			// data from flash

reg		[31:0]	iter;				// iterator for SPI state

reg		[7:0]	STATE;				// current master state
reg		[7:0]	NEXT_STATE;			// next state after idle

reg		ERROR;						// interla error flag (should be 0)

reg		[7:0]	sclc_cnt;
reg		SCLK;

// reg		[7:0]	coder_data			// register to code data to SPI interface
// reg		[7:0]	decoder_data		// register to decoded data from SPI
// reg		ready;						// SPI ready flag

always @(posedge CLK) begin
	if (RESET) begin
		sclc_cnt <= 0;
		SCLK <= 0;
	end else if (sclc_cnt == 49) begin
		SCLK <= ~SCLK;
		sclc_cnt <= 0;
	end else begin
		sclc_cnt <= sclc_cnt + 1;
	end
end

// =============================================================================
always @(posedge SCLK or posedge RESET)
	if (RESET) begin
		STATE		<= IDLE;
		NEXT_STATE	<= FLASH_ISNTR_READ;
		// coder_data	<= READ_DATA_INSTR
		flash_instr	<= READ_DATA_INSTR;
		// decoder_data<= 7'd0;
		FLASH_CS	<= 1'd1;
		ERROR		<= 1'd0;
	end
	
	else begin
	case (STATE)

// =============================================================================
/*
 * first task - read 4 bytes from flash memory to `flash_data` register
 */
		IDLE: begin
			STATE			<= NEXT_STATE;
			iter			<= 31'd0;
			FLASH_CS		<= 1'd0;
		end

		// ---------------------------------------------------------------------
		FLASH_ISNTR_READ: begin
			MOSI			<= flash_instr[iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd7) begin
				STATE		<= FLASH_ADDR_READ;
				flash_addr	<= READ_ADDR;
				// coder_data	<= READ_ADDR[23:16];
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_ADDR_READ: begin
			MOSI			<= flash_addr[23 - iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd23) begin
				STATE		<= FLASH_READ_STATE;
				iter		<= 31'd0;
				// coder_data	<= 7'd0;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_READ_STATE: begin
			MOSI			<= 1'd0;
			flash_data[iter]<= MISO;
			iter			<= iter + 31'd1;
			if (iter == 31'd63) begin
				STATE		<= LED_GLOW_1;
				FLASH_CS	<= 1'd1;
				iter		<= 31'd0;
			end
		end

// =============================================================================
/*
 * second task - show red data on 8 seven segment LEDs
 */
		LED_GLOW_1: begin
			MOSI			<= flash_data[31'd63 - iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd63) begin
				NEXT_STATE	<= FLASH_WRITE_ENABLE;
				STATE		<= IDLE;
				FLASH_CS	<= 1'd1;
				flash_instr	<= WRITE_ENABLE_INSTR;
			end
		end

// =============================================================================
/*
 * third task - write red 4 bytes back to flash memory from `flash_data`
 * register
 */
		FLASH_WRITE_ENABLE: begin
			MOSI			<= flash_instr[iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd7) begin
				NEXT_STATE	<= FLASH_INSTR_WRITE;
				STATE		<= IDLE;
				FLASH_CS	<= 1'd1;
				flash_instr	<= WRITE_ENABLE_INSTR;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_INSTR_WRITE: begin
			MOSI			<= flash_instr[iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd7) begin
				STATE		<= FLASH_ADDR_WRITE;
				flash_addr	<= WRITE_ADDR;
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_ADDR_WRITE: begin
			MOSI			<= flash_addr[23 - iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd23) begin
				STATE		<= FLASH_WRITE_STATE;
				FLASH_CS	<= 1'd1;
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_WRITE_STATE: begin
			MOSI			<= flash_data[iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd63) begin
				NEXT_STATE	<= FLASH_WRITE_DISABLE;
				STATE		<= IDLE;
				FLASH_CS	<= 1'd1;
				flash_instr	<= WRITE_DISABLE_INSTR;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_WRITE_DISABLE: begin
			MOSI			<= flash_instr[iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd7) begin
				NEXT_STATE	<= FLASH_INSTR_FAST;
				STATE		<= IDLE;
				FLASH_CS	<= 1'd1;
				flash_instr	<= FAST_READ_INSTR;
			end
		end

// =============================================================================
/*
 * fourth task - read 4 bytes from flash memory to `flash_data` register using
 * `Fast Read` command
 */
		FLASH_INSTR_FAST: begin
			MOSI			<= flash_instr[iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd7) begin
				STATE		<= FLASH_ADDR_FAST;
				flash_addr	<= READ_ADDR_FAST;
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_ADDR_FAST: begin
			MOSI			<= flash_addr[23 - iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd23) begin
				STATE		<= FLASH_FAST_DUMMY;
				FLASH_CS	<= 1'd1;
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_FAST_DUMMY: begin
			iter			<= iter + 31'd1;
			if (iter == 31'd7) begin
				STATE		<= FLASH_FAST_READ;
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		FLASH_FAST_READ: begin
			MOSI			<= 1'd0;
			flash_data[iter]<= MISO;
			iter			<= iter + 31'd1;
			if (iter == 31'd63) begin
				STATE		<= LED_GLOW_2;
				FLASH_CS	<= 1'd1;
				iter		<= 31'd0;
			end
		end

// =============================================================================
/*
 * fifth task - show red data on 8 seven segment LEDs
 */
		LED_GLOW_2: begin
			MOSI			<= flash_data[31'd63 - iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd63) begin
				STATE		<= SENSOR_ISNTR_READ;
				SENSOR_CS	<= 1'd0;
			end
		end

// =============================================================================
/*
 * sixth task - read data from sensor to `flash_data` registor
 */
		SENSOR_ISNTR_READ: begin
			MOSI			<= flash_instr[iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd7) begin
				STATE		<= SENSOR_ADDR_READ;
				flash_addr	<= READ_ADDR;
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		SENSOR_ADDR_READ: begin
			MOSI			<= flash_addr[23 - iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd23) begin
				STATE		<= SENSOR_READ_STATE;
				iter		<= 31'd0;
			end
		end

		// ---------------------------------------------------------------------
		SENSOR_READ_STATE: begin
			MOSI			<= 1'd0;
			flash_data[iter]<= MISO;
			iter			<= iter + 31'd1;
			if (iter == 31'd63) begin
				STATE		<= LED_GLOW_3;
				SENSOR_CS	<= 1'd1;
				iter		<= 31'd0;
			end
		end


// =============================================================================
/*
 * seventh task - show red data on 8 seven segment LEDs
 */
		LED_GLOW_3: begin
			MOSI			<= flash_data[31'd63 - iter];
			iter			<= iter + 31'd1;
			if (iter == 31'd63) begin
				STATE		<= TERMINATE;
			end
		end

// =============================================================================
/*
 * end of the program
 */
		TERMINATE: begin
			STATE		<= TERMINATE;
		end

		default: begin
			ERROR		<= 1'd1;
			STATE		<= TERMINATE;
		end

	endcase
end

endmodule
