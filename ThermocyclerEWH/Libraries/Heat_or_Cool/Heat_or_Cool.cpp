/*
  Heat_or_Cool.cpp - Library for either heating or cooling the AL block.
  Created by Adam Li, November 9th, 2013.
  Released into public domain for EWH usage.
*/

#include "Arduino.h"		//include Arduino function controls
#include "Heat_or_Cool.h"	//include heating/cooling library
#include "LiquidCrystal.h"	//include LCD screen library

// Constructor /////////////////////////////////////////////////////////////////
// Function that handles the creation and setup of instances


Heat_or_Cool :: Heat_or_Cool(int cw, int ccw)
{
  //initialize library
  pinMode(cw, OUTPUT);
  pinMode(ccw, OUTPUT);  
  
  //initialize variables
  _cw = cw;
  _ccw = ccw;

}

/*
%  Switches direction of motor driver to heating
%  Function Prototype: void heat()
%  
%  Input: void
%  Output: makes peltier heat up
*/
void Heat_or_Cool::heat(void)
{
  //alg. to heat top of peltier
  digitalWrite(_cw, HIGH);
  digitalWrite(_ccw, LOW);
  

}

/*
%  Switches direction of motor driver to cooling
%  Function Prototype: void Cool()
%  
%  Input: void
%  Output: makes peltier cool down
*/
void Heat_or_Cool::cool(void)
{
  //alg. to cool top of peltier
  digitalWrite(_cw, LOW);
  digitalWrite(_ccw, HIGH);

}
