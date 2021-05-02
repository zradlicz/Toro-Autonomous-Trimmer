void updateSerial() {
  static char serInput[GCODE_MAXCHARS + 1] = "";
  static size_t numInBuffer = 0;
  static bool noWaiting = false;
  char* newline;
  
  numInBuffer += Serial.readBytes(serInput + numInBuffer, GCODE_MAXCHARS - numInBuffer);
  serInput[numInBuffer] = '\0';

  if(noWaiting 
     && !gcodeBuffer.full() 
     && Serial.available() < GCODE_MAXCHARS - numInBuffer) {
      Serial.println("waiting");
      noWaiting = false;
    }
  
  newline = strchr(serInput, '\n');
  if(newline != NULL) {
    #ifdef DEBUGG
    Serial.print("Got full line >");
    Serial.print(serInput);
    Serial.println("<");
    mainTimer.print();
    mainTimer.reset();
    #endif

    if(!gcodeBuffer.full() && Serial.available() < GCODE_MAXCHARS - numInBuffer) {
      Serial.println("waiting");
    } else {
      noWaiting = true;
    }
    while(newline != NULL) {
      parseLineToGcode(serInput, newline);
      newline = strchr(serInput, '\n');
    }
    
    numInBuffer = strlen(serInput);
  } else if(numInBuffer == GCODE_MAXCHARS) {
    Serial.println(F("serial buffer filled but no newline, clearing buffer"));
    serInput[0] = '\0';
    numInBuffer = 0;
  }
}

/* turns this:
 * one two three\nfour five\0
 * ^str         ^strEnd
 * into this:
 * four five\0ree\0four five\0
 * ^str          ^strEnd
 */
void parseLineToGcode(char* str, char* strEnd) {
  const char s[] = " ";
  char* token;
  bool validGcode = true;

  gcodeBuffer.setPrev();

  *strEnd = '\0';
  /* get the first token */
  token = strtok(str, s);
  /* walk through other tokens */
  while( token != NULL ) {
    #ifdef DEBUGG
    Serial.println(token);
    #endif
    validGcode &= tokenToGcode(token);
    token = strtok(NULL, s);
  }

  if(validGcode) {
    gcodeBuffer.push();
  } else {
    Serial.println("Invalid gcode command");
  }
  
  strcpy(str, strEnd+1);
}

bool tokenToGcode(char* token) {
  char code = token[0];
  switch (code) {
    case 'G': // tokens which set a letter and an int
    case 'M':
    case 'S':
    case 'F':
    case 'R':
    case 'P':
      gcodeBuffer.setFunc(atoi(token+1));
    case '?': // token which only sets a letter
      gcodeBuffer.setLetter(code);
      break;
    case 'X': // tokens which only set a float
    case 'Y':
    case 'Z':
      gcodeBuffer.setCoord(code,atof(token+1));
      break;
    default:
      return false;
  }
  return true;
}
