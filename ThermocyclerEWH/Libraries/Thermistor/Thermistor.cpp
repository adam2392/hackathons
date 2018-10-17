/*
 * Thermistor.cpp - Library for calculating the temperature of the AL block based on 	                 thermistor.
 * based on resistance changes.
 * 
 * Created by Adam Li, November 23, 2013.
 * Released into public domain for EWH usage
*/

#include "Arduino.h"		//include Arduino function controls
#include "Thermistor.h"		//include Thermistor library

// Constructor /////////////////////////////////////////////////////////////////
// Function that handles the creation and setup of instances

Thermistor :: Thermistor(float A, float B, float C, float pad)
{
  //initialize variables
  _A = A;
  _B = B;
  _C = C;
  _pad = pad;
}

/*
%  Function Prototype: float temp()
%  Calculates temperature of thermistor and uses the Steinhart-Hart
%  Equation. 
%
%  Input: analog voltage
%  Output: returns the temperature reading of thermistor
*/

float Thermistor::temp(int RawADC)
{ 
  float temp;
  float rt;
  rt = log((1024 * _pad / RawADC) - _pad); //initial thermistor resistance calculation
  temp = 1/(_A + _B*rt + _C*rt*rt*rt);	    //temperature calculation
  temp = temp - 273.15;			    //convert kelvin to celcius
  return temp;
}

