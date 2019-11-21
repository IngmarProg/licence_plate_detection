// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//
#pragma on
#define TESTING // antud juhul v�etakse globaalsest muutujast, millist faili avada

#ifdef TESTING
	#define COUTENABLED
#endif

// #define DEBUG // siis kuvatakse k�iki aknaid jne
#define _USETESSERACT




// #define DARKCOLOR 95
#define WHITECOLOR_RANGESTART 0x9B9B9B
#define WHITECOLOR_RANGEEND 0xFFFFFF

#define DARKCOLOR_RANGESTART 0x888888
#define DARKCOLOR_RANGEEND 0x000000

// nr pildi suurus; kui pilt l�hedalt tehtud, siis sobib ideaalselt
#define DEFAULT_RATIOFNR 4.6272
#define DEFAULT_DELTACORR 1.2

// siin laseme pildi ratio kordades l�dvemaks, rohkem vigu, aga t�en�osus, et miskit leitakse on suurem

#define DEFAULT_RATIOFNR_LOOSE 4.1
#define DEFAULT_DELTACORR_LOOSE 2.3

//#define DEFAULT_RATIOFNR_LOOSE 5.1
//#define DEFAULT_DELTACORR_LOOSE 3.4

#define SHOW_OCR_PREPARED_IMG // Kuvatakse pilt, mis l�heb OCRile anal��simiseks
//#define SHOW_FOUNDNUMBER  // N�itab l�htepilti, kus regioon punaseks v�rvitud, kus midagi leiti

// p�him�tteliselt tegemist tulevikus automaattestimise reziimiga; testimises vaid debug lubatud ja cout !
#ifndef TESTING
	#undef DEBUG
	#undef COUTENABLED	
#endif //  TESTING


#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: reference additional headers your program requires here
