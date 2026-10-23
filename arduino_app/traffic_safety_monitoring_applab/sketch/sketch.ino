// UNO Q MCU sketch. Ordinary small LEDs only, each with a series resistor.
// Red: D7 -> 1 kohm -> anode; green: D8 -> 1 kohm -> anode.
// Both cathodes -> GND. Do not connect either anode to the 5V rail.
#include <Arduino_RouterBridge.h>
#include <atomic>

constexpr int RED_PIN = 7;
constexpr int GREEN_PIN = 8;
constexpr unsigned long WATCHDOG_MS = 1500;
std::atomic<int> requestedState{0};
std::atomic<unsigned long> lastCommand{0};

int zone_set_leds(int state) {
  if (state < 0 || state > 2) return -1;
  // Bridge callbacks may execute on another thread. Only loop writes GPIO.
  lastCommand.store(millis());
  requestedState.store(state);
  return state;
}

void setup() {
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  Bridge.begin();
  Bridge.provide("zone_set_leds", zone_set_leds);
}

void loop() {
  static int applied = -1;
  int state = requestedState.load();
  if (millis() - lastCommand.load() > WATCHDOG_MS) state = 0;
  if (state != applied) {
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, LOW);
    if (state == 1) digitalWrite(RED_PIN, HIGH);
    if (state == 2) digitalWrite(GREEN_PIN, HIGH);
    applied = state;
  }
  delay(10);
}
