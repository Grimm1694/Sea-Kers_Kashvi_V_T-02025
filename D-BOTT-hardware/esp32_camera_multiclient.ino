/*

  This is a simple MJPEG streaming webserver implemented for AI-Thinker ESP32-CAM
  and ESP-EYE modules.
  This is tested to work with VLC and Blynk video widget and can support up to 10
  simultaneously connected streaming clients.
  Simultaneous streaming is implemented with FreeRTOS tasks.

  Inspired by and based on this Instructable: $9 RTSP Video Streamer Using the ESP32-CAM Board
  (https://www.instructables.com/id/9-RTSP-Video-Streamer-Using-the-ESP32-CAM-Board/)

  Board: AI-Thinker ESP32-CAM or ESP-EYE
  Compile as:
   ESP32 Dev Module
   CPU Freq: 240
   Flash Freq: 80
   Flash mode: QIO
   Flash Size: 4Mb
   Patrition: Minimal SPIFFS
   PSRAM: Enabled
*/

// ESP32 has two cores: APPlication core and PROcess core (the one that runs ESP32 SDK stack)
#define APP_CPU 1
#define PRO_CPU 0

#include "src/OV2640.h"
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiClient.h>
#include <ESPmDNS.h>


#include <esp_bt.h>
#include <esp_wifi.h>
#include <esp_sleep.h>
#include <driver/rtc_io.h>

// Select camera model
//#define CAMERA_MODEL_WROVER_KIT
// #define CAMERA_MODEL_ESP_EYE
//#define CAMERA_MODEL_M5STACK_PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE
#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"

  const int led = 33; // GPIO for the onboard LED to be controlled
  const int flash = 4; // GPIO for the flash LED

  // +++ Define pins for D13 and D14 +++
  const int D13_PIN = 13;
  const int D14_PIN = 14;
  // ++++++++++++++++++++++++++++++++++++


/*
  Next one is an include with wifi credentials.
  This is what you need to do:

  1. Create a file called "home_wifi_multi.h" in the same folder   OR   under a separate subfolder of the "libraries" folder of Arduino IDE. (You are creating a "fake" library really - I called it "MySettings").
  2. Place the following text in the file:
  #define SSID1 "replace with your wifi ssid"
  #define PWD1 "replace your wifi password"
  3. Save.

  Should work then
*/
#include "home_wifi_multi.h"

OV2640 cam;

WebServer server(80);

// ===== rtos task handles =========================
// Streaming is implemented with 3 tasks:
TaskHandle_t tMjpeg;   // handles client connections to the webserver
TaskHandle_t tCam;     // handles getting picture frames from the camera and storing them locally
TaskHandle_t tStream;  // actually streaming frames to all connected clients

// frameSync semaphore is used to prevent streaming buffer as it is replaced with the next frame
SemaphoreHandle_t frameSync = NULL;

// Queue stores currently connected clients to whom we are streaming
QueueHandle_t streamingClients;

// We will try to achieve 25 FPS frame rate
const int FPS = 14;

// We will handle web client requests every 50 ms (20 Hz)
const int WSINTERVAL = 100;


// ======== Server Connection Handler Task ==========================
void mjpegCB(void* pvParameters) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(WSINTERVAL);

  // Creating frame synchronization semaphore and initializing it
  frameSync = xSemaphoreCreateBinary();
  xSemaphoreGive( frameSync );

  // Creating a queue to track all connected clients
  streamingClients = xQueueCreate( 10, sizeof(WiFiClient*) );

  //=== setup section  ==================

  //  Creating RTOS task for grabbing frames from the camera
  xTaskCreatePinnedToCore(
    camCB,        // callback
    "cam",        // name
    4096,         // stacj size
    NULL,         // parameters
    2,            // priority
    &tCam,        // RTOS task handle
    APP_CPU);     // core

  //  Creating task to push the stream to all connected clients
  xTaskCreatePinnedToCore(
    streamCB,
    "strmCB",
    4 * 1024,
    NULL, //(void*) handler,
    2,
    &tStream,
    APP_CPU);

  //  Registering webserver handling routines
  server.on("/mjpeg/1", HTTP_GET, handleJPGSstream);
  server.on("/jpg", HTTP_GET, handleJPG);

  // +++ Add handlers for LED control (GPIO 33) +++
  server.on("/led/on", HTTP_GET, handleLedOn);
  server.on("/led/off", HTTP_GET, handleLedOff);
  // +++++++++++++++++++++++++++++++++++++++++++++

  // +++ Add handlers for D13 control +++
  server.on("/v1/on", HTTP_GET, handleD13On);
  server.on("/v1/off", HTTP_GET, handleD13Off);
  // ++++++++++++++++++++++++++++++++++++

  // +++ Add handlers for D14 control +++
  server.on("/v2/on", HTTP_GET, handleD14On);
  server.on("/v2/off", HTTP_GET, handleD14Off);
  // ++++++++++++++++++++++++++++++++++++

  // +++ Add handlers for Flash control (GPIO 4) +++
  server.on("/flash/on", HTTP_GET, handleFlashOn);
  server.on("/flash/off", HTTP_GET, handleFlashOff);
  // +++++++++++++++++++++++++++++++++++++++++++++++

  server.onNotFound(handleNotFound);

  //  Starting webserver
  server.begin();

  //=== loop() section  ===================
  xLastWakeTime = xTaskGetTickCount();
  for (;;) {
    server.handleClient();

    //  After every server client handling request, we let other tasks run and then pause
    taskYIELD();
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}


// Commonly used variables:
volatile size_t camSize;    // size of the current frame, byte
volatile char* camBuf;      // pointer to the current frame


// ==== RTOS task to grab frames from the camera =========================
void camCB(void* pvParameters) {

  TickType_t xLastWakeTime;

  //  A running interval associated with currently desired frame rate
  const TickType_t xFrequency = pdMS_TO_TICKS(1000 / FPS);

  // Mutex for the critical section of swithing the active frames around
  portMUX_TYPE xSemaphore = portMUX_INITIALIZER_UNLOCKED;

  //  Pointers to the 2 frames, their respective sizes and index of the current frame
  char* fbs[2] = { NULL, NULL };
  size_t fSize[2] = { 0, 0 };
  int ifb = 0;

  //=== loop() section  ===================
  xLastWakeTime = xTaskGetTickCount();

  for (;;) {

    //  Grab a frame from the camera and query its size
    cam.run();
    size_t s = cam.getSize();

    //  If frame size is more that we have previously allocated - request  125% of the current frame space
    if (s > fSize[ifb]) {
      fSize[ifb] = s * 4 / 3;
      if (fSize[ifb] == 0) { // Safety check for zero size
          Serial.println("Warning: Calculated frame buffer size is 0. Setting to a minimum.");
          fSize[ifb] = 1024; // Arbitrary small size to prevent issues, adjust as needed
      }
      fbs[ifb] = allocateMemory(fbs[ifb], fSize[ifb]);
    }

    //  Copy current frame into local buffer
    char* b = (char*) cam.getfb();
    if (b != NULL && s > 0 && s <= fSize[ifb]) { // Add check for valid buffer and size
        memcpy(fbs[ifb], b, s);
    } else if (b == NULL) {
        Serial.println("cam.getfb() returned NULL");
        // Potentially skip frame or handle error
        vTaskDelayUntil(&xLastWakeTime, xFrequency); // Wait for next cycle
        continue;
    } else if (s == 0) {
        Serial.println("cam.getSize() returned 0");
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
        continue;
    } else if (s > fSize[ifb]) {
        Serial.printf("Frame size %d > buffer size %d. This should have been handled by reallocation.\n", s, fSize[ifb]);
        // This indicates a logic error or rapid frame size increase not caught by reallocation logic
        // Consider reallocating immediately or handling the error
        fSize[ifb] = s * 4 / 3; // Attempt re-allocation again
        fbs[ifb] = allocateMemory(fbs[ifb], fSize[ifb]);
        if (s <= fSize[ifb]) { // Check if reallocation was successful
            memcpy(fbs[ifb], b, s);
        } else {
            Serial.println("Reallocation failed to provide enough space. Skipping frame.");
            vTaskDelayUntil(&xLastWakeTime, xFrequency);
            continue;
        }
    }


    //  Let other tasks run and wait until the end of the current frame rate interval (if any time left)
    taskYIELD();
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    //  Only switch frames around if no frame is currently being streamed to a client
    //  Wait on a semaphore until client operation completes
    xSemaphoreTake( frameSync, portMAX_DELAY );

    //  Do not allow interrupts while switching the current frame
    portENTER_CRITICAL(&xSemaphore);
    camBuf = fbs[ifb];
    camSize = s;
    ifb++;
    ifb &= 1;  // this should produce 1, 0, 1, 0, 1 ... sequence
    portEXIT_CRITICAL(&xSemaphore);

    //  Let anyone waiting for a frame know that the frame is ready
    xSemaphoreGive( frameSync );

    //  Technically only needed once: let the streaming task know that we have at least one frame
    //  and it could start sending frames to the clients, if any
    xTaskNotifyGive( tStream );

    //  Immediately let other (streaming) tasks run
    taskYIELD();

    //  If streaming task has suspended itself (no active clients to stream to)
    //  there is no need to grab frames from the camera. We can save some juice
    //  by suspedning the tasks
    if ( eTaskGetState( tStream ) == eSuspended ) {
      vTaskSuspend(NULL);  // passing NULL means "suspend yourself"
    }
  }
}


// ==== Memory allocator that takes advantage of PSRAM if present =======================
char* allocateMemory(char* aPtr, size_t aSize) {

  //  Since current buffer is too smal, free it
  if (aPtr != NULL) free(aPtr);

  if (aSize == 0) {
    Serial.println("allocateMemory called with aSize = 0. Returning NULL.");
    return NULL; // Or handle as an error, but don't try to malloc(0)
  }

  size_t freeHeap = ESP.getFreeHeap();
  char* ptr = NULL;

  // If memory requested is more than 2/3 of the currently free heap, try PSRAM immediately
  if ( aSize > freeHeap * 2 / 3 ) {
    if ( psramFound() && ESP.getFreePsram() >= aSize ) { // Use >= for size check
      ptr = (char*) ps_malloc(aSize);
      if(ptr) Serial.printf("Allocated %d bytes from PSRAM (strategy 1)\n", aSize); else Serial.printf("Failed to allocate %d from PSRAM (strategy 1)\n", aSize);
    } else {
        Serial.printf("Not enough PSRAM or PSRAM not found. Requested: %d, Free PSRAM: %d\n", aSize, ESP.getFreePsram());
    }
  }
  else {
    //  Enough free heap - let's try allocating fast RAM as a buffer
    ptr = (char*) malloc(aSize);
    if(ptr) Serial.printf("Allocated %d bytes from HEAP\n", aSize); else Serial.printf("Failed to allocate %d from HEAP\n", aSize);

    //  If allocation on the heap failed, let's give PSRAM one more chance:
    if ( ptr == NULL && psramFound() && ESP.getFreePsram() >= aSize) { // Use >= for size check
      ptr = (char*) ps_malloc(aSize);
      if(ptr) Serial.printf("Allocated %d bytes from PSRAM (strategy 2)\n", aSize); else Serial.printf("Failed to allocate %d from PSRAM (strategy 2)\n", aSize);
    } else if (ptr == NULL) {
        Serial.printf("Heap alloc failed & (PSRAM not found or not enough PSRAM). Requested: %d, Free PSRAM: %d\n", aSize, ESP.getFreePsram());
    }
  }

  // Finally, if the memory pointer is NULL, we were not able to allocate any memory, and that is a terminal condition.
  if (ptr == NULL) {
    Serial.printf("!!! Memory allocation failed for size %d. FreeHeap: %d, FreePSRAM: %d. Restarting.\n", aSize, freeHeap, ESP.getFreePsram());
    delay(100); // Give serial a moment to send
    ESP.restart();
  }
  return ptr;
}


// ==== STREAMING ======================================================
const char HEADER[] = "HTTP/1.1 200 OK\r\n" \
                      "Access-Control-Allow-Origin: *\r\n" \
                      "Content-Type: multipart/x-mixed-replace; boundary=123456789000000000000987654321\r\n";
const char BOUNDARY[] = "\r\n--123456789000000000000987654321\r\n";
const char CTNTTYPE[] = "Content-Type: image/jpeg\r\nContent-Length: ";
const int hdrLen = strlen(HEADER);
const int bdrLen = strlen(BOUNDARY);
const int cntLen = strlen(CTNTTYPE);


// ==== Handle connection request from clients ===============================
void handleJPGSstream(void)
{
  //  Can only acommodate 10 clients. The limit is a default for WiFi connections
  if ( !uxQueueSpacesAvailable(streamingClients) ) {
    server.send(503, "text/plain", "Too many clients"); // Send a 503 Service Unavailable
    return;
  }


  //  Create a new WiFi Client object to keep track of this one
  WiFiClient* client = new WiFiClient();
  *client = server.client();

  //  Immediately send this client a header
  client->write(HEADER, hdrLen);
  client->write(BOUNDARY, bdrLen);

  // Push the client to the streaming queue
  xQueueSend(streamingClients, (void *) &client, 0);

  // Wake up streaming tasks, if they were previously suspended:
  if ( tCam != NULL && eTaskGetState( tCam ) == eSuspended ) vTaskResume( tCam );
  if ( tStream != NULL && eTaskGetState( tStream ) == eSuspended ) vTaskResume( tStream );
}


// ==== Actually stream content to all connected clients ========================
void streamCB(void * pvParameters) {
  char buf[16];
  TickType_t xLastWakeTime;
  TickType_t xFrequency;

  //  Wait until the first frame is captured and there is something to send
  //  to clients
  ulTaskNotifyTake( pdTRUE,          /* Clear the notification value before exiting. */
                    portMAX_DELAY ); /* Block indefinitely. */

  xLastWakeTime = xTaskGetTickCount();
  for (;;) {
    // Default assumption we are running according to the FPS
    xFrequency = pdMS_TO_TICKS(1000 / FPS);

    //  Only bother to send anything if there is someone watching
    UBaseType_t activeClients = uxQueueMessagesWaiting(streamingClients);
    if ( activeClients ) {
      // Adjust the period to the number of connected clients
      if (activeClients > 0) { // Prevent division by zero if activeClients becomes 0 unexpectedly
          xFrequency /= activeClients;
      }
      if (xFrequency == 0) xFrequency = 1; // Ensure xFrequency is not zero


      //  Since we are sending the same frame to everyone,
      //  pop a client from the the front of the queue
      WiFiClient *client;
      xQueueReceive (streamingClients, (void*) &client, 0);

      //  Check if this client is still connected.
      if (client == NULL) continue; // Should not happen with proper queue usage

      if (!client->connected()) {
        //  delete this client reference if s/he has disconnected
        //  and don't put it back on the queue anymore. Bye!
        delete client;
      }
      else {

        //  Ok. This is an actively connected client.
        //  Let's grab a semaphore to prevent frame changes while we
        //  are serving this frame
        if (xSemaphoreTake( frameSync, pdMS_TO_TICKS(100) ) == pdTRUE) { // Wait max 100ms
            if (camBuf != NULL && camSize > 0) { // Check if frame data is valid
                client->write(CTNTTYPE, cntLen);
                sprintf(buf, "%d\r\n\r\n", camSize);
                client->write(buf, strlen(buf));
                client->write((char*) camBuf, (size_t)camSize);
                client->write(BOUNDARY, bdrLen);
                client->flush(); 
            } else {
                Serial.println("streamCB: camBuf is NULL or camSize is 0. Skipping send.");
            }
            //  The frame has been served. Release the semaphore
            xSemaphoreGive( frameSync );
            
            // Since this client is still connected, push it to the end
            // of the queue for further processing
            xQueueSend(streamingClients, (void *) &client, 0);
        } else {
            Serial.println("streamCB: Timeout waiting for frameSync semaphore. Re-queueing client.");
            // Could not get semaphore, requeue client to try again later
            xQueueSend(streamingClients, (void *) &client, 0);
        }
        taskYIELD();
      }
    }
    else {
      //  Since there are no connected clients, there is no reason to waste battery running
      Serial.println("streamCB: No active clients, suspending stream task.");
      vTaskSuspend(NULL);
    }
    //  Let other tasks run after serving every client
    taskYIELD();
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}



const char JHEADER[] = "HTTP/1.1 200 OK\r\n" \
                       "Content-disposition: inline; filename=capture.jpg\r\n" \
                       "Content-type: image/jpeg\r\n\r\n";
const int jhdLen = strlen(JHEADER);

// ==== Serve up one JPEG frame =============================================
void handleJPG(void)
{
  WiFiClient client = server.client();

  if (!client.connected()) return;
  
  // Capture a fresh frame. This needs to be protected if cam.run() is not reentrant
  // or if it modifies shared resources used by camCB.
  // Given camCB also calls cam.run(), it's safer to use the buffered frame
  // or ensure cam.run() can be called independently here without messing with camCB's fb.
  // For simplicity, let's use the latest frame from camCB.
  // If an absolutely fresh frame is needed, cam.run() would be called, but ensure OV2640.cpp
  // handles internal frame buffer management correctly for such ad-hoc captures.

  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;

  // Attempt to take a new picture directly
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed for /jpg");
    server.send(500, "text/plain", "Failed to capture image");
    return;
  }

  if (fb->format != PIXFORMAT_JPEG) {
    Serial.println("Non-JPEG format not supported for /jpg");
    esp_camera_fb_return(fb); // IMPORTANT: return the frame buffer
    server.send(500, "text/plain", "Non-JPEG format captured");
    return;
  }

  client.write(JHEADER, jhdLen);
  client.write(fb->buf, fb->len);
  
  esp_camera_fb_return(fb); // IMPORTANT: return the frame buffer
}

// +++ LED Control Handlers (GPIO 33) +++
void handleLedOn() {
  digitalWrite(led, HIGH);
  server.send(200, "text/plain", "LED on GPIO 33 is ON");
  Serial.println("LED (33) ON command received");
}

void handleLedOff() {
  digitalWrite(led, LOW);
  server.send(200, "text/plain", "LED on GPIO 33 is OFF");
  Serial.println("LED (33) OFF command received");
}
// ++++++++++++++++++++++++++++++++++++++

// +++ D13 Control Handlers (GPIO 13) +++
void handleD13On() {
  digitalWrite(D13_PIN, HIGH);
  server.send(200, "text/plain", "GPIO 13 is ON");
  Serial.println("GPIO 13 ON command received");
}

void handleD13Off() {
  digitalWrite(D13_PIN, LOW);
  server.send(200, "text/plain", "GPIO 13 is OFF");
  Serial.println("GPIO 13 OFF command received");
}
// ++++++++++++++++++++++++++++++++++++++

// +++ D14 Control Handlers (GPIO 14) +++
void handleD14On() {
  digitalWrite(D14_PIN, HIGH);
  server.send(200, "text/plain", "GPIO 14 is ON");
  Serial.println("GPIO 14 ON command received");
}

void handleD14Off() {
  digitalWrite(D14_PIN, LOW);
  server.send(200, "text/plain", "GPIO 14 is OFF");
  Serial.println("GPIO 14 OFF command received");
}
// ++++++++++++++++++++++++++++++++++++++

// +++ Flash Control Handlers (GPIO 4) +++
void handleFlashOn() {
  digitalWrite(flash, HIGH);
  server.send(200, "text/plain", "Flash (GPIO 4) is ON");
  Serial.println("Flash (4) ON command received");
}

void handleFlashOff() {
  digitalWrite(flash, LOW);
  server.send(200, "text/plain", "Flash (GPIO 4) is OFF");
  Serial.println("Flash (4) OFF command received");
}
// +++++++++++++++++++++++++++++++++++++++


// ==== Handle invalid URL requests ============================================
void handleNotFound()
{
  String message = "Server is running!\n\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET) ? "GET" : "POST";
  message += "\nArguments: ";
  message += server.args();
  message += "\n\nAvailable endpoints:\n";
  message += "/mjpeg/1 - MJPEG stream\n";
  message += "/jpg - Single JPEG capture\n";
  message += "/led/on - Turn LED (GPIO 33) ON\n";
  message += "/led/off - Turn LED (GPIO 33) OFF\n";
  message += "/v1/on - Turn GPIO 13 ON\n";
  message += "/v1/off - Turn GPIO 13 OFF\n";
  message += "/v2/on - Turn GPIO 14 ON\n";
  message += "/v2/off - Turn GPIO 14 OFF\n";
  message += "/flash/on - Turn Flash (GPIO 4) ON\n";
  message += "/flash/off - Turn Flash (GPIO 4) OFF\n";
  server.send(200, "text/plain", message);
}



// ==== SETUP method ==================================================================
void setup()
{
  // Setup Serial connection:
  Serial.begin(115200);
  // Wait for serial to initialize
  // (Host computer may need a few seconds to recognize the new serial port)
  unsigned long startTime = millis();
  while(!Serial && millis() - startTime < 2000); // Wait max 2 seconds for serial
  
  Serial.println("\n\nStarting setup...");

  pinMode(led , OUTPUT);
  pinMode(flash, OUTPUT);   
  digitalWrite(led, LOW); // Ensure LED is off initially
  digitalWrite(flash, LOW); // Ensure flash is off initially

  // +++ Initialize D13 and D14 pins +++
  pinMode(D13_PIN, OUTPUT);
  digitalWrite(D13_PIN, LOW); // Ensure D13 is off initially
  pinMode(D14_PIN, OUTPUT);
  digitalWrite(D14_PIN, LOW); // Ensure D14 is off initially
  Serial.println("GPIO 33, 4, 13, 14 initialized as OUTPUT LOW.");
  // ++++++++++++++++++++++++++++++++++++


  // Configure the camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Frame parameters: pick one
  //  config.frame_size = FRAMESIZE_UXGA;
  //  config.frame_size = FRAMESIZE_SVGA; // 800x600
   config.frame_size = FRAMESIZE_HD; // 1280x720
  //  config.frame_size = FRAMESIZE_QVGA; // 320x240
  // config.frame_size = FRAMESIZE_VGA; // 640x480
  config.jpeg_quality = 12; //0-63 lower numbers are higher quality
  config.fb_count = 2; // Using 2 frame buffers is good for streaming

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP); // This would conflict with D13_PIN output for ESP_EYE
  pinMode(14, INPUT_PULLUP); // This would conflict with D14_PIN output for ESP_EYE
  Serial.println("ESP_EYE model defined, D13/D14 set as INPUT_PULLUP, may conflict with output controls.");
#endif

  // Initialize the camera
  esp_err_t err = cam.init(config); // Use the OV2640 wrapper's init
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    Serial.println("Make sure camera module is connected properly.");
    Serial.println("Restarting in 10 seconds...");
    delay(10000);
    ESP.restart();
  }
  Serial.println("Camera initialized successfully.");
  
  
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, 1);  // Enable vertical flip (1 = ON, 0 = OFF)
    s->set_hmirror(s, 1); // Enable horizontal mirror
    Serial.println("Camera vflip and hmirror set.");
  } else {
    Serial.println("Failed to get camera sensor for vflip/hmirror!");
  }

  //  Configure and connect to WiFi
  IPAddress ip;

  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID1, PWD1);
  Serial.print("Connecting to WiFi");
  int wifi_retries = 0;
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(F("."));
    digitalWrite(flash , !digitalRead(flash)); // Blink flash LED while connecting
    wifi_retries++;
    if (wifi_retries > 30) { // Timeout after 15 seconds
        Serial.println("\nFailed to connect to WiFi. Check credentials. Restarting...");
        delay(1000);
        ESP.restart();
    }
  }

  digitalWrite(flash , LOW); // Turn off flash LED after connection
  Serial.println("\nWiFi connected!");

  ip = WiFi.localIP();
  Serial.print(F("IP Address: "));
  Serial.println(ip);

  if (!MDNS.begin("esp1")) { 
      Serial.println("Error starting mDNS");
  } else {
      Serial.println("mDNS started successfully! You can access the services at:");
      Serial.print("http://esp2.local/ or http://"); Serial.print(ip); Serial.println("/");
      MDNS.addService("http", "tcp", 80);
  }


  Serial.println("\nAvailable HTTP Endpoints:");
  Serial.print("MJPEG Stream: http://"); Serial.print(ip); Serial.println("/mjpeg/1");
  Serial.print("Single JPG:   http://"); Serial.print(ip); Serial.println("/jpg");
  Serial.print("LED (33) ON:  http://"); Serial.print(ip); Serial.println("/led/on");
  Serial.print("LED (33) OFF: http://"); Serial.print(ip); Serial.println("/led/off");
  Serial.print("GPIO 13 ON:   http://"); Serial.print(ip); Serial.println("/v1/on");
  Serial.print("GPIO 13 OFF:  http://"); Serial.print(ip); Serial.println("/v1/off");
  Serial.print("GPIO 14 ON:   http://"); Serial.print(ip); Serial.println("/v2/on");
  Serial.print("GPIO 14 OFF:  http://"); Serial.print(ip); Serial.println("/v2/off");
  Serial.print("Flash (4) ON: http://"); Serial.print(ip); Serial.println("/flash/on");
  Serial.print("Flash (4) OFF:http://"); Serial.print(ip); Serial.println("/flash/off");


  // Start mainstreaming RTOS task
  xTaskCreatePinnedToCore(
    mjpegCB,
    "mjpegServer", // More descriptive name
    4 * 1024 + 512, // Increased stack for webserver and more handlers
    NULL,
    2, // Priority
    &tMjpeg,
    APP_CPU);

  Serial.println("\nSetup complete. Main server task (mjpegServer) started.");
  Serial.printf("Free Heap: %d, PSRAM: %d\n", ESP.getFreeHeap(), ESP.getFreePsram());
}


void loop() {
  // All work is done in RTOS tasks.
  // This loop can be used for light, non-blocking tasks or just yield.
  vTaskDelay(pdMS_TO_TICKS(1000)); // Yield for 1 second
  // Serial.printf("Loop - Free Heap: %d, PSRAM: %d\n", ESP.getFreeHeap(), ESP.getFreePsram()); // Optional: periodic heap check
}