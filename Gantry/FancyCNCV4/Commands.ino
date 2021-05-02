void rapidPos(float x, float y, float z);
void goHome();

void runG(float x,float y,float z, float r){
  switch (Gfunc){
    case 00:
      //if no move say nothing
      if(YStep.currentPosition() ==(long) (y*SPMM) and ZStep.currentPosition() ==(long) (z*SPMM) and XStep.currentPosition() ==(long) (x*SPMM)){
          break;
      }
      //rapid positioning
      Serial.println("Rapid Positioning to");
      Serial.println(x);
      Serial.println(y);
      Serial.println(z);
      rapidPos(x,y,z);
      break;
      
    case 01:
      //linear interpolation cuts along line
      //if no move say nothing
      if(YStep.currentPosition() ==(long) y*SPMM and ZStep.currentPosition() ==(long) z*SPMM and XStep.currentPosition() ==(long) x*SPMM){
          break;
      }
      Serial.println("Moving Line to");
      Serial.println(x);
      Serial.println(y);
      Serial.println(z);
      linearInterpolate(x,y,z);
      break;

    case 02:
      Serial.println("CW Circular Interpolation To");
      Serial.println(x);
      Serial.println(y);
      Serial.println(z);
      Serial.print("With Radius = ");
      Serial.println(r);
      drawArc(x,y,z,r,true);
      break;

    case 03:
      drawArc(x,y,z,r,false);
      break;
    case 28:
      goHome();
      break;


  }
}
void runM(int MCom){
  switch (MCom){
    case 00:
      //program stop
      break;
    case 30:
      //program return to base and stop
      break;
    case 02:
      //end of program
      break;
    case 03:
      //turn on rotor forward
      Serial.print("Cutting Tool Forward, Speed @ ");
      Serial.println(RotorSpeed);
      analogWrite(LPWM,0);
      analogWrite(RPWM,RotorSpeed);
      break;
    case 04:
      //turn on backward rotor
      Serial.print("Cutting Tool Backward, Speed @ ");
      Serial.println(RotorSpeed);
      analogWrite(RPWM,0);
      analogWrite(LPWM,RotorSpeed);
      break;
    case 05:
      Serial.print("Cutting OFF");
      analogWrite(LPWM,0);
      analogWrite(RPWM,0);
      break;
      
  }
}
