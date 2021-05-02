// The class definition is located in FancyCNCV6
// This class implements a ring buffer of gcode_t structs
// all setXXX functions are for the head node and getXXX are for the tail node

// init function
GcodeBuffer::GcodeBuffer() {
  head = 0;
  tail = 0;
  prev = 0;
  count = 0;

  GcodeBuffer::setHomed();
}

void GcodeBuffer::setLetter(char let) {
  gcode[head].letter = let;
}

void GcodeBuffer::setFunc(byte fu) {
  gcode[head].func = fu;
}

// ASCII characters are actually numbered in ascending order, so 
// 'Z'-'X' = 2 for example
// downside is that this doesn't work when lower and uppercase are mixed
void GcodeBuffer::setCoord(char axis, float val) {
  gcode[head].coord[axis-'X'] = val * COORDMUL;
}

char GcodeBuffer::getLetter() {
  return gcode[tail].letter;
}

byte GcodeBuffer::getFunc() {
  return gcode[tail].func;
}

float GcodeBuffer::getCoord(char axis) {
  return gcode[tail].coord[axis-'X'] / COORDMUL;
}

float GcodeBuffer::getCoord(int axis) {
  return gcode[tail].coord[axis] / COORDMUL;
}

void GcodeBuffer::setPrev() {
  gcode[head].letter = gcode[prev].letter;
  gcode[head].func = gcode[prev].func;
  for(int i=0; i<3; i++) {
    gcode[head].coord[i] = gcode[prev].coord[i];
  }
}

void GcodeBuffer::setHomed() {
  gcode[prev].coord[0] = 0;
  gcode[prev].coord[1] = 0;
  gcode[prev].coord[2] = ZHOME * COORDMUL;
}

void GcodeBuffer::push() {
  int nextHead = (head + 1)%GCODE_BUFFSIZE;
  if(nextHead != tail) {
    prev = head;
    head = nextHead;
    count++;
  } else {
    Serial.println(F("Gcode buffer full; ignoring"));
  }
}

void GcodeBuffer::pop() {
  tail = (tail + 1)%GCODE_BUFFSIZE;
  count--;
}

gcode_t* GcodeBuffer::peek() {
  return &gcode[tail];
}

// returns true if buffer is empty
bool GcodeBuffer::empty() {
  return head==tail;
}

// returns true if buffer is full, maybe should just be count >= GCODE_BUFFSIZE?
bool GcodeBuffer::full() {
  return count >= GCODE_BUFFSIZE - 2;
}
