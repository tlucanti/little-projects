#include "lcd.h"
#include "gpio.h"
#include <lpc214x.h>
#include <stdio.h>

// #include <lpc21xx.h>

#include <lpc214x.h>

typedef unsigned char   uchar;
typedef signed char     schar;
typedef unsigned int    uint;
typedef signed int      sint;

#define bDISPLAY        16

uchar  szHi[bDISPLAY + 8],
       szLo[bDISPLAY + 8];

uchar  const   mpbCyrillic[0x100] =
{
  	0x20,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x20,0x20,0x20,0x20,0x20,0x20,0x20,  // 0
  	0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,  // 1

  	0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,  // 2
  	0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x3B,0x3C,0x3D,0x3E,0x3F,  // 3
  	0x40,0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x4B,0x4C,0x4D,0x4E,0x4F,  // 4
  	0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,0x5B,0x5C,0x5D,0x5E,0x5F,  // 5
  	0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x6B,0x6C,0x6D,0x6E,0x6F,  // 6
  	0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x7B,0x7C,0x7D,0xC5,0x01,  // 7

  	0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,  // 8
  	0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,  // 9
  	0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,  // A

  	0xA2,0xB5,0x20,0x20,0x20,0x20,0x20,0x02,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,  // B
  	0x41,0xA0,0x42,0xA1,0xE0,0x45,0xA3,0xA4,0xA5,0xA6,0x4B,0xA7,0x4D,0x48,0x4F,0xA8,  // C
  	0x50,0x43,0x54,0xA9,0xAA,0x58,0xE1,0xAB,0xAC,0xE2,0xAD,0xAE,0xC4,0xAF,0xB0,0xB1,  // D
  	0x61,0xB2,0xB3,0xB4,0xE3,0x65,0xB6,0xB7,0xB8,0xB9,0xBA,0xBB,0xBC,0xBD,0x6F,0xBE,  // E
  	0x70,0x63,0xBF,0x79,0xE4,0x78,0xE5,0xC0,0xC1,0xE6,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7   // F
};


uchar const mpbUserFonts[64] =
{ 
  	0x1F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  	0x00, 0x16, 0x15, 0x15, 0x0E, 0x04, 0x04, 0x00, 
  	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};


int Sleep (int _slp)
{
	int i, j;
	for (i = 0; i < _slp * 1000; i++)
		j = i + 1;
	return j;
}

void SetCommLCD(uchar  bT)
{  
	IODIR1 |= (0xFF << 16);		// Set data bus as outputs
	IODIR1 |= (1 << 24);	   	// Set RS as output
	IODIR1 |= (1 << 25);		// Set E as output
	IODIR0 |= (1 << 30);		// Set Backlight control as output
	IODIR0 |= (1 << 22);		// Set R/W as output

	IOCLR1 = (1 << 25);			// E down
	IOCLR1 = (1 << 24);			// RS = 0
	IOCLR0 = (1 << 22);			// RW = 0
	IOCLR1 = (0xFF) << 16;		// Clear data bus
	IOSET1 = (bT) << 16;		// Write data to bus
	IOSET1 = (1 << 25);			// E up
	Sleep (10);				   	// Wait
	IOCLR1 = (1 << 25);		   	// E down
}


void SetDataLCD(uchar  bT)
{  
	IODIR1 |= (0xFF << 16);		// Set data bus as outputs
	IODIR1 |= (1 << 24);	   	// Set RS as output
	IODIR1 |= (1 << 25);		// Set E as output
	IODIR0 |= (1 << 30);		// Set Backlight control as output
	IODIR0 |= (1 << 22);		// Set R/W as output

	IOCLR1 = (1 << 25);			// E down
	IOSET1 = (1 << 24);			// RS = 1
	IOCLR0 = (1 << 22);			// RW = 0
	IOCLR1 = (0xFF) << 16;		// Clear data bus
	IOSET1 = (bT) << 16;		// Write data to bus
	IOSET1 = (1 << 25);			// E up
	Sleep (10);					// Wait
	IOCLR1 = (1 << 25);		   	// E down
}                                 

int GetCommLCD(void)
{
	// We should check for Busy flag here
	// Insead of it we just wait...
	// Busy flag can be read via DataBus7 with RS=0, RW=1
	Sleep (1);
	return 0;
}


void  ReadyLCD(void)
{
	while ( GetCommLCD() != 0 );
}


void ShowMsgLCD(uchar  bT, uchar  const *szT)      
{
	uchar   i;

  	ReadyLCD();   SetCommLCD(bT);

  	for (i = 0; i < bDISPLAY; i++)
  	{
    	if ( !*szT ) break;
    	ReadyLCD();      
    	SetDataLCD( mpbCyrillic[ *szT++ ] );
  	}
}

 
                                       
void InitLCD(void)
{
	uchar   i;

  	Sleep (20);   SetCommLCD(0x30);     
  	Sleep (20);   SetCommLCD(0x30);     
  	Sleep (20);   SetCommLCD(0x30);          

  	ReadyLCD();   SetCommLCD(0x38);
  	ReadyLCD();   SetCommLCD(0x08);
  	ReadyLCD();   SetCommLCD(0x01);
  	ReadyLCD();   SetCommLCD(0x06);              
  	ReadyLCD();   SetCommLCD(0x0C);

  	ReadyLCD();   SetCommLCD(0x40);     

  	for (i = 0; i < 64; i++)
  	{
    	ReadyLCD ();
    	SetDataLCD ( mpbUserFonts[i] );
  	}

  	ReadyLCD();   SetCommLCD(0xC4);
}

void LCDTextOut (uchar * top_line, uchar * bottom_line)
{
	ShowMsgLCD(0x80, top_line);
  	ShowMsgLCD(0xC0, bottom_line);
  	ReadyLCD();   SetCommLCD(0xC4);
}

void SetBacklight (int Backlight)
{
  	if (Backlight)
		IOSET0 = 1 << 30;
	else
		IOCLR0 = 1 << 30;
}








int TimerIsUp = 0;

/***************************************************/
/*	            Interrupt handler                  */
/***************************************************/
__irq void Timer0ISR (void)  	//Timer0 ISR
{
	TimerIsUp = 1;              //set Flag
	T0IR = 0x01;                //reset interrupt flag
	VICVectAddr = 0;            //reset VIC
	return;
}
  
void InitTimer0 (void)
{
  	/***************************************************/
	/*	              Init VIC                         */
	/***************************************************/
 	VICDefVectAddr = (unsigned int) &Timer0ISR;
  	VICIntEnable = 0x10;  	// Channel#4 is the Timer0
  	VICIntSelect = 0x00;  	// all interrupts are IRQs
  
  	/***************************************************/
	/*	              Init Timer0                      */
	/***************************************************/
	T0MR0 = 1500000;	    // Timer match (~ 0.1 second)
  	T0MCR = 0x03;        	// Interrupt on Match0, reset timer on match
  	T0PC = 0x01;         	// Prescaler to 2
  	T0TC = 0x00;         	// reset Timer counter
  	T0TCR = 0x01;        	// enable Timer

	/***************************************************/
	/*	              Set flag to zero                 */
	/***************************************************/
	TimerIsUp = 0;

  	return;
}

int GPIOPerformanceCheck (void)
{
	int count = 0;
	int prev_SCS = SCS;
	
	/***************************************************/
	/*	              Init GPIO                        */
	/***************************************************/
	SCS = 0x00;				// Use GPIO (instead of FIO)
	IODIR0 |= 0x0000FF00;  	// Set P0.8 - P0.15 as outputs
	IOSET0 |= 0x0000FF00;	// Clear P0.8 - P0.15 

	InitTimer0 ();			// Start timer

	while (!TimerIsUp)
	{
		// Pull up P0.12
		IOSET0 |= (1 << 12);
		while (!(IOPIN0 & (1 << 12)));	// Wait until P0.12 is up

		// Pull down P0.12
		IOCLR0 |= (1 << 12);
		while (IOPIN0 & (1 << 12));		// Wait until P0.12 is down
		count ++;
	}

	// Restore previous SCS state
	SCS = prev_SCS;

	return count;
}

int FIOPerformanceCheck (void)
{
	int count = 0;
	int prev_SCS = SCS;

	/***************************************************/
	/*	              Init  FIO                        */
	/***************************************************/
	SCS = 0x01;		  			// Use FIO (instead of GPIO)
	FIO0DIR |= 0x0000FF00; 		// Set P0.8 - P0.15 as outputs
	FIO0SET |= 0x0000FF00; 		// Clear P0.8 - P0.15 

	InitTimer0 ();				// Start timer

	while (!TimerIsUp)
	{
		FIO0SET |= (1 << 11);
		while (!(FIO0PIN & (1 << 11)));	 	// Wait until P0.11 is up
		FIO0CLR |= (1 << 11);
		while (FIO0PIN & (1 << 11));		// Wait until P0.11 is down
		count ++;
	}
	
	// Restore previous SCS state
	SCS = prev_SCS;

	return count;
}

void itoa(int n, char *v)
{
	int i = 0;
	int n2 = n;
	while (n2)
	{
		++i;
		n2 /= 10;
	}
	while (n)
	{
		v[i] = n % 10;
		--i;
		n /= 10;
	}
}

#include <string>
int main (void)
{
//	int GPIO, FIO;
//	unsigned char result1 [30];
//	unsigned char result2 [30];
	// unsigned char line1[16];
	// unsigned char line2[16];
	
	/***************************************************/
	/*	       Write Hands On number to LCD            */
	/***************************************************/

	InitLCD();
	SetBacklight(1);
	char *line1 = "    mk sucks    ";
	char *line2 = "    mk sucks    ";
	// std::string l1 = " c++ - cool ";
	// std::string l2 = " mk - sucks ";
	// int i = 0;
	// while (1)
	// {
		// itoa(i, line1); 
	// LCDTextOut((uchar *)l1.c_str(), (uchar *)l2.c_str());

	LCDTextOut((uchar *)line1, (uchar *)line2);		// sleep(1);
	// }
	while (true);

	/***************************************************/
	/*	              Hands On code                    */
	/***************************************************/

	// GPIO = GPIOPerformanceCheck ();
	// sprintf (result1, "GPIO %d ticks", GPIO);
	// FIO = FIOPerformanceCheck ();
	// sprintf (result2, "FIO  %d ticks", FIO);
	// LCDTextOut (result1, result2);

	//while (1);

// 	return 0;
}
