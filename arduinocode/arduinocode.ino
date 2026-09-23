#include <ESP8266WiFi.h>
#include <Servo.h>

const char* ssid = "Y3A";
const char* password = "RCA@2024";

Servo myServo;
WiFiServer server(80);
int currentAngle = 90; // Center position

void setup() {
  Serial.begin(115200);
  myServo.attach(14); // Change D4 to the pin your servo signal wire is connected to
  myServo.write(currentAngle);
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected. IP address: ");
  Serial.println(WiFi.localIP());
  server.begin();
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    String request = client.readStringUntil('\r');
    client.flush();
    
    // Check if the request contains an angle command
    if (request.indexOf("/MOVE?angle=") != -1) {
      int angleStart = request.indexOf("angle=") + 6;
      int angleEnd = request.indexOf(" ", angleStart);
      String angleStr = request.substring(angleStart, angleEnd);
      int angle = angleStr.toInt();
      
      if (angle >= 0 && angle <= 180) {
        currentAngle = angle;
        myServo.write(currentAngle);
      }
    }
    
    // Send HTTP response back to Python
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/html");
    client.println("Connection: close");
    client.println();
    client.println("<!DOCTYPE HTML>");
    client.println("<html><body><h1>Servo Moved to " + String(currentAngle) + "</h1></body></html>");
    delay(1);
  }
}