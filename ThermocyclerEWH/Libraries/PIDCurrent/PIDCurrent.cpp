/*
  PIDCurrent.cpp - Library for changing the current of output PWM based on PID algorithm
  to reach a certain temperature.
  Created by Adam Li, November 23rd, 2013.
  Released into public domain for EWH usage.
  */

#include "Arduino.h"		//include Arduino function controls
#include "Heat_or_Cool.h"	//include heating/cooling library
#include "PID.h"
#include "LiquidCrystal.h"	//include LCD screen library
#include "Thermistor.h"		//include Thermistor library
#include "PIDCurrent.h"
#define out_pwm 3 //pwm for motor driver
#define ThermistorPIN A0 //defining Thermistor Pin on Arduino code


//// variables for Temperature Printing Method
const int D=1; // 0 for printing temperature, 1 to serial output graph

Thermistor thermistor(.000829415622, .000284485, -0.000000117184277, 9800);     //constructor for Thermistor
PID_params par ={80,0,0,0,0,0,7,7,0,95}; //NM:First array cell is the temp setpoint
PID pid = PID(&par);

// Constructor /////////////////////////////////////////////////////////////////
// Function that handles the creation and setup of instances

Heat_or_Cool heat_or_cool(4, 5); //constructor for heat_or_cool


PIDCurrent :: PIDCurrent(int cw, int ccw)
{
  //initialize library
  pinMode(cw, OUTPUT);
  pinMode(ccw, OUTPUT);

  _cw = cw;
  _ccw = ccw;

}

/*
 * A function to change the current output in analogWrite.
 * Function Prototype: void changeCurrent(int temp, longtime1)%  
%  Input: 
%	  int goal_temp is the temperature we want to reach
%	  long time is the time the program has been running
%	  int out_pwm is the pin number we want to write out current amplitude to
%  Output: changes output current of 
*/
void PIDCurrent::changeCurrent(int temp, long time)
{
  long currentAmp = 0; //initialize current amplitude
  long timed; // time for serial output
  int del =500;// delay for serial output
  int row = 0; //for serial output


  
  pid.setInput(temp); //input into PID alg
  pid.process(time);  //calls PID calculation
  currentAmp = pid.getOutput();  //current amp gets output # from PID
  
  //if statements for setting directionality of current (heat/cool)
  if(currentAmp>0){
     heat_or_cool.heat();
     if(currentAmp>255){ 
       currentAmp = 255;
        }
  }
  if(currentAmp<0){
      heat_or_cool.cool();
      if(currentAmp<-255){
    	currentAmp = -255;
  	}
  }

  analogWrite(out_pwm, currentAmp); //sets the pwm to current amplitude

  if (D==0){ //set D=0 if only temperature is wanted
    Serial.print("Celsius: "); 
    Serial.print(temp,1);   // display Celsius
    Serial.println("");
  }
  if (D==1){ //set D=1 if serial output graph is wanted
    if ((millis()-timed)>del){ //makes data be recorded only in increments equal to "del"
      Serial.print("DATA,TIME,"); 
      Serial.println(temp); 
      Serial.println(currentAmp); //puts data in format the PLX-DAQ can interpret
      row++;
      timed=millis();
    }
   }
}

/*
 * A function to change the temperature based on a new set point, and 
 * delay time.
 * Function Prototype: Function Prototype: void reachTemp(int setPoint, long delayHold)
%  Input: 
%	  int goal_temp is the temperature we want to reach
%	  long time is the time the program has been running
%	  int out_pwm is the pin number we want to write out current amplitude to
%  Output: changes temperature of peltier
*/
void PIDCurrent::reachTemp(int setPoint, long delayHold){ // by default flag = 0 is cooling, 1 is heating
  pid.setSetPoint(setPoint); //sets the set point for PID calc
  int currentTemp = thermistor.temp (analogRead(ThermistorPIN));
  int tempgoal = setPoint; //sets our goal as the setpoint temperature
  int flag = currentTemp<tempgoal; //flag is a 0 if cooling, and 1 if heating
  long PIDtime;
  if(flag == 1){ //flag = 1 says we want to heat to our temp goal
	while(currentTemp<tempgoal){ //loops PID until reaches temp goal
		currentTemp= thermistor.temp (analogRead(ThermistorPIN));
  		long PIDtime = millis();
  		changeCurrent(currentTemp, PIDtime); //passes temp and time to change pwm current delivery based on PID alg
	}
  }
  else if(flag == 0){ //flag determines that we want to cool to our temperature goal
	while(currentTemp>tempgoal){ //loops PID until reaches 3rd temp goal
		currentTemp = thermistor.temp (analogRead(ThermistorPIN));
    	        PIDtime = millis();
    	        changeCurrent(currentTemp, PIDtime);
  	}
  }
  unsigned long runningtime = millis(); //get current running time of program
  unsigned long desiredTime = runningtime + delayHold; //running time + 30 second hold
	
  while(runningtime<desiredTime){ //keep temperature at temp goal until delay is reached
	runningtime = millis();
  	currentTemp= thermistor.temp (analogRead(ThermistorPIN));
        PIDtime = millis();
	changeCurrent(currentTemp, PIDtime);
  }
  Serial.print(" temp reached");
}