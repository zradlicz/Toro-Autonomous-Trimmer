void rapidPos(float x, float y, float z) {
  long curX = (long)(x * SPMM);
  long curY = (long)(y * SPMM);
  long curZ = (long)(z * SPMM);
  XStep.moveTo((long)(x * SPMM));
  YStep.moveTo((long)(y * SPMM));
  ZStep.moveTo((long)(z * SPMM));
  XStep.setSpeed(MaxSpeed);
  YStep.setSpeed(MaxSpeed);
  ZStep.setSpeed(MaxSpeed);
  while (true) {
    
    //Add Limits

    XStep.runSpeedToPosition();
    YStep.runSpeedToPosition();
    ZStep.runSpeedToPosition();

    if (YStep.currentPosition() == (long) (y * SPMM) and ZStep.currentPosition() == (long) (z * SPMM) and XStep.currentPosition() == (long) (x * SPMM)) {
      break;
    }
//    if (YStep.currentPosition()-curY<1 and ZStep.currentPosition()-curZ<1 and XStep.currentPosition()-curX<1) {
//      break;
//    }
  }
}
void linearInterpolate(float x, float y, float z) {
  long temp[] = {(long)(x * SPMM), (long)(y * SPMM), (long)(z * SPMM)};

  //rotate to proper cut. 
  
  steppers.moveTo(temp);
  steppers.runSpeedToPosition();
}
void drawArc(float x, float y, float z, float r, bool cw) {
  float curX = ((float) XStep.currentPosition())/SPMM;
  float curY = ((float) YStep.currentPosition())/SPMM;
  float curZ = ((float) ZStep.currentPosition())/SPMM;
  //distance
  
  
  //if the xy coordinates are the same draw a circle
  if (x==curX and y==curY){
    Serial.print("Drawing circle with radius ");
    Serial.println(r);
    //draw circle with radius r around current point
    rapidPos(x,y+r,curZ);
    //rotate cutter to right 90 degrees and turn on
    rapidPos(x,y+r,z);
    if(cw){
      for (int i =0;i<=360; i++){
        linearInterpolate(r*sin(i*6.28/360)+x,r*cos(i*6.28/360)+y,z);
      }
    }
    else{
      for (int i =360;i>=0; i+=-1){
        linearInterpolate(r*sin(i*6.28/360)+x,r*cos(i*6.28/360)+y,z);
      }
    }
    rapidPos(x,y,curZ);
    return;
  }
  //if they aren't the same, do what??? probably the partial arc code goes here.
  float xydist = sqrt(sq(x-curX)+sq(y-curY));
  if (xydist > 2*r){
    Serial.println("Arc Impossible");
    return;
  }
  else{
    //find centerpoint
    
  }

  
  return;
}

void rotateThetaTo(float theta){
  //prevent wrapping otherwise go to closest
  float curTheta = ((float) ThetaStep.currentPosition())/SPREV*360;

  //find closest theta
  //if(
  //}
  //verify it doesn't wrap to much
  
  ThetaStep.moveTo(SPREV*theta/360);
  ThetaStep.setSpeed(THETASPEED);
  ThetaStep.runSpeedToPosition();
}

void goHome(){
  Serial.println("Homing");
  bool x = !digitalRead(XLIMSWITCH);
  bool y = !digitalRead(YLIMSWITCH);
  XStep.setSpeed(-MaxSpeed);
  YStep.setSpeed(-MaxSpeed);
  while(true){
    if(x and y){
      Serial.print("Home Sweet Home (x=0,y=0)");
      YStep.setCurrentPosition(0);
      XStep.setCurrentPosition(0);
      break;
    }
    if(!x){
      XStep.runSpeed();
      x = !digitalRead(XLIMSWITCH);
    }
    if(!y){
      YStep.runSpeed();
      y = !digitalRead(YLIMSWITCH);
    }
  }
}
