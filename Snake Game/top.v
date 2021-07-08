
`timescale 1ns / 1ps
////////////////////////////////////////////////////////////////////////////////
// Company: MIET
// Engineer: CHEL
// 
// Create Date: 01.04.2021 11:15:38
// Design Name: 
// Module Name: top
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

`ifndef TOP_H
`define TOP_H
`include "hvsync_generator.v"
`include "divider.v"
`include "divider"

module switches_top(clk, reset, hsync, vsync, switches_p1, switches_p2, rgb);

	input	[7:0]	switches_p1;
	input	[7:0]	switches_p2;
	input			clk;
	input			reset;
	output			hsync;
	output			vsync;
	output	[2:0]	rgb;

// -----------------------------------------------------------------------------
	parameter		head_horiz_initial1	= GRID_SIZE;  // ball initial X position
	parameter		head_vert_initial1	= GRID_SIZE; // ball initial Y position
	parameter		head_horiz_initial2	= GRID_SIZE * (grid_cnt - 2);  // ball initial X position
	parameter		head_vert_initial2	= GRID_SIZE * (grid_cnt - 3); // ball initial Y position
	parameter		GRID_SIZE			= 15;   // ball size (in pixels)
	parameter		SPEED				= 3;
	parameter		SNAKE_SIZE			= 5;
	
	localparam		grid_cnt			= 256 / GRID_SIZE;

// -----------------------------------------------------------------------------
	wire			up1					= switches_p1[2];
	wire			down1				= switches_p1[3];
	wire			left1				= switches_p1[0];
	wire			right1				= switches_p1[1];
	wire			up2					= switches_p2[2];
	wire			down2				= switches_p2[3];
	wire			left2				= switches_p2[0];
	wire			right2				= switches_p2[1];
	wire			borders				= display_on && (hpos < 2 || vpos < 2 || hpos > 254 || vpos > 238);
	wire			r_out				= r_grid | rh1 | rh2 | r1 | r2 | apple_gfx;
	wire			g_out				= g_grid | gh1 | gh2 | g1 | g2;
	wire			b_out				= b_grid | bh1 | bh2 | b1 | b2 | borders;
	wire			display_on;
	wire			sync;

// -----------------------------------------------------------------------------
	reg		[8:0]	hpos;
	reg		[8:0]	vpos;
	reg		[8:0]	apple_h;
	reg		[8:0]	apple_v;
	reg		[8:0]	head_h1;
	reg		[8:0]	head_v1;
	reg		[8:0]	head_h2;
	reg		[8:0]	head_v2;

	reg		[8:0]	head_hdiff1;
	reg		[8:0]	head_vdiff1;
	reg		[8:0]	head_hdiff2;
	reg		[8:0]	head_vdiff2;
	reg		[8:0]	apple_hdiff;
	reg		[8:0]	apple_vdiff;
	reg		[8:0]	apple_h_wire;
	reg		[8:0]	apple_v_wire;
	reg		[8:0]	head_horiz_move1	= GRID_SIZE;
	reg		[8:0]	head_vert_move1		= 0;
	reg		[8:0]	head_horiz_move2	= GRID_SIZE;
	reg		[8:0]	head_vert_move2		= 0;
	reg		[8:0]	size_reg			= GRID_SIZE;

	reg		[8:0]	tailh1				[SNAKE_SIZE-1 : 0];
	reg		[8:0]	tailv1				[SNAKE_SIZE-1 : 0];
	reg		[8:0]	tail_hdiff1 			[SNAKE_SIZE-1 : 0];
	reg		[8:0]	tail_vdiff1 			[SNAKE_SIZE-1 : 0];
	reg				tail_hgfx1			[SNAKE_SIZE-1 : 0];
	reg				tail_vgfx1			[SNAKE_SIZE-1 : 0];
	reg				tail_gfx1			[SNAKE_SIZE-1 : 0];
	reg				tail_r1				[SNAKE_SIZE-1 : 0];
	reg				tail_g1				[SNAKE_SIZE-1 : 0];
	reg				tail_b1				[SNAKE_SIZE-1 : 0];
	reg		[SNAKE_SIZE-1 : 0]		engaged_tail1;//		[SNAKE_SIZE-1 : 0];
	reg				head_hgfx1;
	reg				head_vgfx1;
	reg				head_gfx1;
	reg		[8:0]	tailh2				[SNAKE_SIZE-1 : 0];
	reg		[8:0]	tailv2				[SNAKE_SIZE-1 : 0];
	reg		[8:0]	tail_hdiff2			[SNAKE_SIZE-1 : 0];
	reg		[8:0]	tail_vdiff2			[SNAKE_SIZE-1 : 0];
	reg				tail_hgfx2			[SNAKE_SIZE-1 : 0];
	reg				tail_vgfx2			[SNAKE_SIZE-1 : 0];
	reg				tail_gfx2			[SNAKE_SIZE-1 : 0];
	reg				tail_r2				[SNAKE_SIZE-1 : 0];
	reg				tail_g2				[SNAKE_SIZE-1 : 0];
	reg				tail_b2				[SNAKE_SIZE-1 : 0];
	reg		[SNAKE_SIZE-1 : 0]		engaged_tail2;//		[SNAKE_SIZE-1 : 0];
	reg				head_hgfx2;
	reg				head_vgfx2;
	reg				head_gfx2;
	reg				grid_gfx;
	reg				apple_hgfx;
	reg				apple_vgfx;
	reg				apple_gfx;
	reg				r_grid;
	reg				g_grid;
	reg				b_grid;
	reg				rh1;
	reg				gh1;
	reg				bh1;
	reg				r1;
	reg				g1;
	reg				b1;
	reg				rh2;
	reg				gh2;
	reg				bh2;
	reg				r2;
	reg				g2;
	reg				b2;
	reg				true;
	reg				game_over;
 
// -----------------------------------------------------------------------------
  	genvar			i0;
	genvar			i1;
	genvar			i2;
	genvar			i3;
	genvar			i4;
	genvar			i5;
	genvar			i6;

// -----------------------------------------------------------------------------

	hvsync_generator hvsync_gen(
		.clk(clk),
		.reset(reset),
		.hsync(hsync),
		.vsync(vsync),
		.display_on(display_on),
		.hpos(hpos),
		.vpos(vpos)
	);
	
	divider #(
		.speed(SPEED)
	) div (
		.in(vsync),
		.out(sync)
	);
	
	random # (
		.min(1),
		.max(grid_cnt),
		.seed(0),
		.n(9)
	) rand_1 (
		.clk(sync),
		.num(apple_h_wire)
	);

	random # (
		.min(1),
		.max(grid_cnt),
		.seed(1),
		.n(9)
	) rand_2 (
		.clk(sync),
		.num(apple_v_wire)
	);

// -----------------------------------------------------------------------------
	always @(posedge sync or posedge reset) begin
		if (reset) begin
			for (i5=0; i5 < SNAKE_SIZE; i5 = i5 + 1) begin
				engaged_tail1[i5] <= 1'b0;
				engaged_tail2[i5] <= 1'b0;
			end
			engaged_tail1[0] <= 1'b1;
			engaged_tail1[1] <= 1'b1;
			engaged_tail2[0] <= 1'b1;
			engaged_tail2[1] <= 1'b1;
			apple_h <= GRID_SIZE * (grid_cnt / 2);
			apple_v <= GRID_SIZE * (grid_cnt / 2);
			true	<= 1'b1;
		end
		if (head_h1 + head_horiz_move1 == apple_h
				&& head_v1 + head_vert_move1 == apple_v) begin
			apple_h <= apple_h_wire * GRID_SIZE;
			apple_v <= apple_v_wire * GRID_SIZE;
			engaged_tail1 <= {engaged_tail1[SNAKE_SIZE-2:0], 1'b1};
		end
		if (head_h2 + head_horiz_move2 == apple_h
				&& head_v2 + head_vert_move2 == apple_v) begin
			apple_h <= apple_h_wire * GRID_SIZE;
			apple_v <= apple_v_wire * GRID_SIZE;
			engaged_tail2 <= {engaged_tail2[SNAKE_SIZE-2:0], 1'b1};
		end
	end

	always @(posedge sync or posedge reset)
	begin
		if (reset) begin
			head_h1				<= head_horiz_initial1;
			head_v1				<= head_vert_initial1;
			head_h2				<= head_horiz_initial2;
			head_v2				<= head_vert_initial2;
			for (i0 = 0; i0 < SNAKE_SIZE; i0=i0+1) begin
				tailh1[i0]		<= head_horiz_initial1 - GRID_SIZE * (1);
				tailv1[i0]		<= head_vert_initial1;
				tailh2[i0]		<= head_horiz_initial2 + GRID_SIZE * (1);
				tailv2[i0]		<= head_vert_initial2;
			end
		end
		else if (~game_over) begin
			head_h1				<= head_h1 + head_horiz_move1;
			head_v1				<= head_v1 + head_vert_move1;
			head_h2				<= head_h2 + head_horiz_move2;
			head_v2				<= head_v2 + head_vert_move2;
			for (i1 = 1; i1 < SNAKE_SIZE; i1=i1+1) begin
				tailh1[i1]		<= tailh1[i1 - 1];
				tailv1[i1]		<= tailv1[i1 - 1];
				tailh2[i1]		<= tailh2[i1 - 1];
				tailv2[i1]		<= tailv2[i1 - 1];
			end
			tailh1[0]			<= head_h1;
			tailv1[0]			<= head_v1;
			tailh2[0]			<= head_h2;
			tailv2[0]			<= head_v2;
		end
	end

	always @(posedge left1 or posedge right1 or posedge up1
		or posedge down1 or posedge reset or posedge left2 or posedge right2 or posedge up2
		or posedge down2) begin
		if (reset | (right1 && head_horiz_move1 != -size_reg)) begin
			head_horiz_move1	<= size_reg;
			head_vert_move1		<= 0;
		end
		else if (left1 && head_horiz_move1 != size_reg) begin
			head_horiz_move1		<= -size_reg;
			head_vert_move1		<= 0;
		end
		else if (up1 && head_vert_move1 != size_reg) begin
			head_horiz_move1		<= 0;
			head_vert_move1		<= -size_reg;
		end
		else if (down1 && head_vert_move1 != -size_reg) begin
			head_horiz_move1		<= 0;
			head_vert_move1		<= size_reg;
		end
		if (right2 && head_horiz_move2 != -size_reg) begin
			head_horiz_move2		<= size_reg;
			head_vert_move2		<= 0;
		end
		else if (reset | (left2 && head_horiz_move2 != size_reg)) begin
			head_horiz_move2		<= -size_reg;
			head_vert_move2		<= 0;
		end
		else if (up2 && head_vert_move2 != size_reg) begin
			head_horiz_move2		<= 0;
			head_vert_move2		<= -size_reg;
		end
		else if (down2 && head_vert_move2 != -size_reg) begin
			head_horiz_move2		<= 0;
			head_vert_move2		<= size_reg;
		end
	end

	always @(posedge clk or posedge reset) begin
 		if (reset) begin
 			for (i4=0; i4 < SNAKE_SIZE; i4 = i4 + 1) begin
				tail_hdiff1[i4]	<= 0;
				tail_vdiff1[i4]	<= 0;
				tail_hgfx1[i4]	<= 0;
				tail_vgfx1[i4]	<= 0;
				tail_gfx1 [i4]	<= 0;
				tail_r1[i4]		<= 0;
				tail_g1[i4]		<= 0;
				tail_b1[i4]		<= 0;
				tail_hdiff2[i4]	<= 0;
				tail_vdiff2[i4]	<= 0;
				tail_hgfx2[i4]	<= 0;
				tail_vgfx2[i4]	<= 0;
				tail_gfx2 [i4]	<= 0;
				tail_r2[i4]		<= 0;
				tail_g2[i4]		<= 0;
				tail_b2[i4]		<= 0;
			end
			size_reg <= GRID_SIZE;
 		end
		for (i2 = 0; i2 < SNAKE_SIZE; i2 = i2 + 1) begin
			tail_hdiff1[i2]		<= hpos - tailh1[i2];
			tail_vdiff1[i2]		<= vpos - tailv1[i2] - 3;
			tail_hgfx1[i2]		<= tail_hdiff1[i2] < GRID_SIZE - 4;
			tail_vgfx1[i2]		<= tail_vdiff1[i2] < GRID_SIZE - 4;
			tail_gfx1[i2]		<= tail_hgfx1[i2] && tail_vgfx1[i2];
			tail_hdiff2[i2]		<= hpos - tailh2[i2];
			tail_vdiff2[i2]		<= vpos - tailv2[i2] - 3;
			tail_hgfx2[i2]		<= tail_hdiff2[i2] < GRID_SIZE - 4;
			tail_vgfx2[i2]		<= tail_vdiff2[i2] < GRID_SIZE - 4;
			tail_gfx2[i2]		<= tail_hgfx2[i2] && tail_vgfx2[i2];
		end
		head_hdiff1				<= hpos - head_h1 + 1;
		head_vdiff1				<= vpos - head_v1 - 1;
		head_hdiff2				<= hpos - head_h2 + 1;
		head_vdiff2				<= vpos - head_v2 - 1;
		apple_hdiff				<= hpos - apple_h;
		apple_vdiff				<= vpos - apple_v - 1;
		head_hgfx1				<= head_hdiff1 < GRID_SIZE - 1;
		head_vgfx1				<= head_vdiff1 < GRID_SIZE - 1;
		head_hgfx2				<= head_hdiff2 < GRID_SIZE - 1;
		head_vgfx2				<= head_vdiff2 < GRID_SIZE - 1;
		apple_hgfx				<= apple_hdiff < GRID_SIZE - 1;
		apple_vgfx				<= apple_vdiff < GRID_SIZE - 1;
		head_gfx1				<= head_hgfx1 && head_vgfx1;
		grid_gfx				<= (((hpos % GRID_SIZE)==0)
			& ((vpos % GRID_SIZE)==0));
		head_gfx2				<= head_hgfx2 && head_vgfx2;
		apple_gfx				<= apple_hgfx && apple_vgfx;
		r_grid					<= 0;
		g_grid					<= display_on & grid_gfx;
		b_grid					<= display_on & grid_gfx;
		rh1						<= 0;
		gh1						<= display_on & (head_gfx1);
		bh1						<= 0;
		rh2						<= 0;
		gh2						<= display_on & (head_gfx2);
		bh2						<= display_on & (head_gfx2);

		for (i3=0; i3 < SNAKE_SIZE; i3 = i3 + 1) begin
			tail_r1[i3]			<= display_on && tail_gfx1[i3] && engaged_tail1[i3];
			tail_g1[i3]			<= display_on && tail_gfx1[i3] && engaged_tail1[i3];
			tail_b1[i3]			<= 0;
			tail_r2[i3]			<= display_on && tail_gfx2[i3] && engaged_tail2[i3];
			tail_g2[i3]			<= 0;
			tail_b2[i3]			<= display_on && tail_gfx2[i3] && engaged_tail2[i3];
		end
	end

	always @* begin
		r1						= 1'b0;
		g1						= 1'b0;
		b1						= 1'b0;
		r2						= 1'b0;
		g2						= 1'b0;
		b2						= 1'b0;
		game_over				= 1'b0;
		for (i3=0; i3 < SNAKE_SIZE; i3 = i3 + 1) begin
			r1					= r1 || tail_r1[i3];
			g1					= g1 || tail_g1[i3];
			b1					= b1 || tail_b1[i3];
			r2					= r2 || tail_r2[i3];
			g2					= g2 || tail_g2[i3];
			b2					= b2 || tail_b2[i3];
			game_over			= game_over
				|| (head_h1 == tailh1[i3] && head_v1 == tailv1[i3])
				|| (head_h2 == tailh2[i3] && head_v2 == tailv2[i3])
				|| head_h1 > 256 || head_h2 > 256
				|| head_v1 > 240 || head_v2 > 240;
		end
	end
	assign	rgb					= {game_over | b_out,g_out,r_out};

endmodule

`endif
