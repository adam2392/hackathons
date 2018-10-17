/*
 * Thermistor.h - Library for calculating the temperature of the AL block based on thermistor.
 * based on resistance changes.
 * 
 * Created by Adam Li, November 23, 2013.
 * Released into public domain for EWH usage
 */

#ifndef Thermistor_h
#define Thermistor_h

#include "Arduino.h"

class Thermistor
{
   public:
	Thermistor (float A, float B, float C, float pad);
	float temp(int RawADC);	//calculates the temperature and outputs

   private:
	int _RawADC;
	float _pad;
	float _A;
	float _B;
	float _C;
};

#endif