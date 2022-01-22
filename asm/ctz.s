j test

cto:
# ------------------------------------------------------------------------------
# cto function - counting trailing ones function
# parameters:
#   a0 - number to count ones from
# return value:
#   a0 - computed value
# using registers:
#   a0, a1, a2, a7, t0
# tlucanti (c)

	not		a0	a0				# invert bits to count trailing ones as zeros
	mv		a7	ra				# saving return address to a7 (because ctz dont
								# use a7 register)
	call	ctz					# call ctz with inverted a0 argument
	mv		ra	a7				# restore return address
	ret							# return

ctz:
# ------------------------------------------------------------------------------
# ctz function - counting trailing zeros function
# parameters:
#   a0 - number to count zeros from
# return value:
#   a0 - computed value
# using registers:
#   a0, a1, a2, t0
# tlucanti (c)

	mv		a1	a0				# a1 = ~a0
	li		a0	1				# init return value a0 = 1

	## first bin search step - checking ones in 16 lsb bits

	li		t0	0x0000FFFF		# magic number to find ones in 16 lsb bits
	and		a2	a1	t0			# | a2 = 16 if input has only zeros in 16 lsb
	seqz	a2	a2				# | bits else a2 = 0
	slli	a2	a2	4			# |

	add		a0	a0	a2			# | add to answer 16 or 0 depending on a2
								# | condition
	srl		a1	a1	a2			# | next searching in 8 lsb or [23:16] bits
								# | depending on a2

	## second bin search step - checking odd or even bytes of word

	andi	a2	a1	0x000000FF	# | a2 = 8 if input has only zeros in 8 lsb
	seqz	a2	a2				# | bits else a2 = 0
	slli	a2	a2	3			# |

	add		a0	a0	a2			# add to answer 8 or 0 depending on a2 condition
	srl		a1	a1	a2			# | next searching in left or riht part of half
								# | word (left or right 8 bits)

	## thrd bin search step - checking odd or even half bytes of word

	andi	a2	a1	0x0000000F	# | a2 = 4 if input has only zeros in 4 lsb bits
	seqz	a2	a2				# | bits else a2 = 0
	slli	a2	a2	2			# |

	add		a0	a0	a2			# add to answer 4 or 0 depending on a2 condition
	srl		a1	a1	a2			# | next searching in left or riht part of
								# | quarter word (left or right 4 bits)

	## fourth bin search step - checking odd or even half bytes of word

	andi	a2	a1	0x00000003	# | a2 = 2 if input has only zeros in 2 lsb bits
	seqz	a2	a2				# | bits else a2 = 0
	slli	a2	a2	1			# |

	add		a0	a0	a2			# add to answer 2 or 0 depending on a2 condition
	srl		a1	a1	a2			# | next searching in left or riht part bit of
								# | remaining part

	## last bin search step - checking odd or event bits of word

	andi	a2	a1	0x00000001	# | a2 = 1 if input has lsb bit is 1
								# | else a2 is 0
	sub		a0	a0	a2			# | substructing a2 from answer to balance
								# | initial 1 value
	ret

test:
	li		a0	0
	call	cto
	mv		s0	a0
	# answer is 0
	
	li		a0	0xFFFFFFFF
	call	cto
	mv		s1	a0
	# answer is 31
	
	li		a0	0x00FF
	call	cto
	mv		s2	a0
	# answer is 8
	
	li		a0	0x000000F
	call	cto
	mv		s3	a0
	# answer is 4
	
	li		a0	0x0FFFFFFF
	call	cto
	mv		s4	a0
	# answer is 28
	
	li		a0	0b10101010101010101010101010101010
	call	cto
	mv		s5	a0
	# answer is 0
	
	li		a0	0b01010101010101010101010101010101
	call	cto
	mv		s6	a0
	# answer is 1
	
	li		a0	0b00010101110000011100000000000000
	call	cto
	mv		s7	a0
	# answer is 0
	
	li		a0	0b10101110000011100000000000001111
	call	cto
	mv		s8	a0
	# answer is 4
	
	li		a0	0x1c211b15
	call	cto
	mv		s9	a0
	# answer is 1
	
	li		a0	0xe33b0985
	call	cto
	mv		s10	a0
	# answer is 1
	
	li		a0	0xdfde4899
	call	cto
	mv		s11	a0
	# answer is 1
	
