/*
@file arduino_motor_driver.h
@brief Motor device driver for the Arduino motor shield.
@author Adam Li
					A			B
Direction			Digital 12	Digital 13
Speed (PWM)		Digital 3		Digital 11
Brake			Digital 9		Digital 8
Current Sensing	Analog 0		Analog 1
*/

#include <motor_driver.h>
#include <Arduino.h>

namespace Leebot
{
	class Motor : public MotorDriver
	{
	public:
		int currentSpeed;		//the current pwm pin
		int pwmPin;			//pin to control pwm current
		int motorPin;			//pin to control motor direction
		int brakePin;			//pin to control brake

		/*
		@brief The Class constructor.
		@param number the DC motor number to control from 1 to 2 ( A or B )
		*/
		Motor(int number) : MotorDriver(), currentSpeed(0)
		{
			if(number == 1) //A
			{
				motorPin = 12;
				pwmPin = 3;
				brakePin = 9;

				digitalWrite(motorPin, HIGH);
				digitalWrite(brakePin, LOW);
			}
			else if(number == 2) //B
			{
				motorPin = 13;
				pwmPin = 11;
				brakePin = 8;

				digitalWrite(motorPin, HIGH);
				digitalWrite(brakePin, LOW);
			}
		}
		
		/*Function: setSpeed
		@brief sets the speed of the DC motor
		*/
		void setSpeed(int speed)
		{
			currentSpeed = speed;
			if (speed >= 0) {
				digitalWrite(motorPin, LOW);
				analogWrite(pwmPin, speed);
			}
			else {
				digitalWrite(motorPin, HIGH);
				analogWrite(pwmPin, speed);
			}
		}

		/*Function: getSpeed()
		@brief returns the current speed from -255 to 255.
		*/
		int getSpeed() const
		{
			return currentSpeed;
		}
	};
};
