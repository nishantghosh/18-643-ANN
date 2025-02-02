/* fread example: read an entire file */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define BYTE_SIZE 8
#define MAX_SHIFT 24

#define META_DETA_START 0
#define META_DATA_END 16
#define LABEL_END 8

#define IMAGE_SIZE 784
#define TRAIN_NUM_IMAGES 20000
#define DEV_NUM_IMAGES 5000
#define NUM_NEURONS 128
#define NUM_OUTPUTS 10
#define NUM_ITERATIONS 15
#define LEARNING_RATE 0.9
#define BRAM_SIZE 4.92 //MB
#define POLY_MASK_32 0xB4BCD35C
#define POLY_MASK_31 0x7A5BC2E3

float init = 0.01;
int lfsr32 = 0xABCDE;
int lfsr31 = 0x23456789;


void print_image(uint8_t img[IMAGE_SIZE]);
void print_layer_1(float img[NUM_NEURONS]);
void print_layer_2(float img[NUM_OUTPUTS]);

int compute_first_layer(int, int);
int compute_second_layer();

int compute_errors(int);
int compute_deltas(int);
float compute_MSE(int);

int parse_images(int);
int parse_labels(int);

int find_answer(float img[NUM_OUTPUTS]);
void init_weights();

void print_weights_1(float w[IMAGE_SIZE]);
void print_weights_2(float w[NUM_NEURONS]);

float sigmoid(float);

uint8_t train_images [TRAIN_NUM_IMAGES][IMAGE_SIZE]; 
uint8_t dev_images   [DEV_NUM_IMAGES][IMAGE_SIZE];
uint8_t train_labels [TRAIN_NUM_IMAGES];
uint8_t dev_labels   [DEV_NUM_IMAGES];

float   weights1     [NUM_NEURONS][IMAGE_SIZE];
float   weights2     [NUM_OUTPUTS][NUM_NEURONS];

float   layer1       [NUM_NEURONS];
float   layer2       [NUM_OUTPUTS];

float   errors_layer1 [NUM_NEURONS];
float   errors_layer2 [NUM_OUTPUTS];

#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[32;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"


int find_answer(float layer2[NUM_OUTPUTS]){

  int max_index = 0;
  float max = layer2[0];
  int iteration;

  for (iteration = 0; iteration < NUM_OUTPUTS; iteration++){
    if (layer2[iteration] > max){
      max = layer2[iteration];
      max_index = iteration;
    }
  }
  return max_index;
}


float sigmoid(float x){


  /* Plan Approximation */

  float return_value;

  if ((fabs(x) >= 0) && (fabs(x) < 1)){
    return_value = 0.25*fabs(x) + 0.5;
  } 
  else if ((fabs(x) >= 1) && (fabs(x) < 2.375)){
    return_value = 0.125*fabs(x) + 0.625;
  }
  else if ((fabs(x) >= 2.375) && (fabs(x) < 5)){
    return_value = 0.03125*fabs(x) + 0.84375;
  }
  else {
    return_value = 1.0;
  }

  if (x < 0)
    return_value = 1.0 - return_value;

  return return_value;
}


int main () {

  parse_images(0); //Train
  parse_labels(0);

  parse_images(1);
  parse_labels(1);

  int img;
  int answer;
  int correct = 0;
  int iteration = 0;
  float MSE = 0;
  float total_MB = 0;
  float space_for_images = 0;

  total_MB += (NUM_NEURONS + NUM_OUTPUTS) * 4.0 * 2.0;
  total_MB += (NUM_NEURONS * NUM_OUTPUTS) * 4.0;
  total_MB += (IMAGE_SIZE  * NUM_NEURONS) * 4.0; 

  total_MB = total_MB/(1024.0 * 1024.0);

  printf(CYAN_TEXT("Mem without images = %f MB\n"), total_MB); 
  
  space_for_images = (float)BRAM_SIZE - total_MB;

  printf(CYAN_TEXT("No of Images = %f\n"), (space_for_images * 1024.0 * 1024.0)/(IMAGE_SIZE + 1));

  init_weights();
  
  for (iteration =0; iteration < NUM_ITERATIONS; iteration++){

    MSE = 0;

    for (img = 0; img < TRAIN_NUM_IMAGES; img++){
      compute_first_layer(0, img);
      compute_second_layer();
 
      compute_errors(img);
 
      compute_deltas(img);
      MSE += compute_MSE(img);
    }

    MSE = MSE/TRAIN_NUM_IMAGES;
    printf("%d. MSE = %f\n", iteration + 1, MSE);
  }

  for (img = 0; img < DEV_NUM_IMAGES; img++){
    compute_first_layer(1, img);
    compute_second_layer();
    answer = find_answer(layer2);

    //print_image(dev_images[img]);
   
    if (answer == dev_labels[img]){
      correct += 1;
    }
  }

  printf(CYAN_TEXT("ACCURACY = %f%\n"), (float)(correct * 100.0)/(float)DEV_NUM_IMAGES);

  return 0;
}


int compute_deltas(int img){

  int output = 0;
  int neuron = 0;
  int iteration = 0;

  for (output = 0; output < NUM_OUTPUTS; output++){
    for (neuron = 0; neuron < NUM_NEURONS; neuron++){
      weights2[output][neuron] += LEARNING_RATE*errors_layer2[output]*layer1[neuron]; 
    }
  }
  
  for (neuron = 0; neuron < NUM_NEURONS; neuron++){
    for (iteration = 0; iteration < IMAGE_SIZE; iteration++){
      weights1[neuron][iteration] += (LEARNING_RATE * errors_layer1[neuron])*((float)(train_images[img][iteration])/255.0);
    }
  }
}


float compute_MSE(int img){

  float MSE = 0;
  int output = 0;
  float truth;

  for (output = 0; output < NUM_OUTPUTS; output++){
    if (output == train_labels[img])
      truth = 1.0;
    else
      truth = 0.0;
  
    MSE += (truth - layer2[output]) * (truth - layer2[output]);
  }

  return MSE;
}

int compute_errors(int img){

  int output = 0;
  int neuron = 0;
  float truth;

  for (output = 0; output < NUM_OUTPUTS; output++){

    if (output == train_labels[img])
      truth = 1.0;
    else 
      truth = 0.0;
    
    errors_layer2[output] = (1.0 - (layer2[output])) * layer2[output] * (truth - layer2[output]);
  }
  
  for (neuron = 0; neuron < NUM_NEURONS; neuron++){

    float contribution = 0;
    for (output = 0; output < NUM_OUTPUTS; output++){
      contribution += weights2[output][neuron]*errors_layer2[output];
    }
    errors_layer1[neuron] = (1.0 - (layer1[neuron])) * layer1[neuron] * contribution;
  }

  return 0;
}


int shift_lfsr(int * lfsr, int polynomial_mask){

  int feedback;
  
  feedback = *lfsr & 1;
  *lfsr >>= 1;
  if (feedback == 1){
    *lfsr ^= polynomial_mask;
  }

  return *lfsr;
}

int get_random(void){

  shift_lfsr(&lfsr32, POLY_MASK_32);
  return (shift_lfsr(&lfsr32, POLY_MASK_32) ^ shift_lfsr(&lfsr31, POLY_MASK_31)) & 0xffff;

}


void init_weights(){

  
  int iteration;
  int neuron;
  int output;

  for (neuron = 0; neuron < NUM_NEURONS; neuron++){
    for (iteration = 0; iteration < IMAGE_SIZE; iteration++){
      weights1[neuron][iteration]  = (((float)get_random()/(float)(RAND_MAX))*init) - init/2.0;
    }
  }

  for (output = 0; output < NUM_OUTPUTS; output++){
    for (iteration = 0; iteration < NUM_NEURONS; iteration++){
      weights2[output][iteration] =  (((float)get_random()/(float)(RAND_MAX))*init) - init/2.0;
    }
  }
}

int compute_first_layer(int dev, int image_no){

  int total_images;
  int iteration;
  int neuron;

  for (neuron = 0; neuron < NUM_NEURONS; neuron++){

    layer1[neuron] = 0;
    iteration = 0;
    while (iteration < IMAGE_SIZE){

      if (dev){
        layer1[neuron] += (float)((dev_images[image_no][iteration])*(weights1[neuron][iteration]/255.0));
      }
      else{
        layer1[neuron] += (float)((train_images[image_no][iteration])*(weights1[neuron][iteration]/255.0));
      }

      iteration += 1;
    }

    layer1[neuron] = sigmoid(layer1[neuron]);
  }

  return 0;
}


int compute_second_layer(){

  int neuron;
  int output;

  for (output = 0; output < NUM_OUTPUTS; output++){

    layer2[output] = 0;
    neuron = 0;
    while (neuron < NUM_NEURONS){

      layer2[output] += (float)((layer1[neuron])*(weights2[output][neuron]));
      neuron += 1;
    }

    layer2[output] = sigmoid(layer2[output]);
  }
  return 0;
}



int parse_labels(int dev){

  if (dev)
    printf(RED_TEXT("Parsing development labels\n"));
  else
    printf(RED_TEXT("Parsing training labels\n"));

  FILE * pFile;
  uint8_t read_byte;
  int magic_no   = 0;
  int num_images = 0;

  if (dev)
    pFile = fopen("t10k-labels.idx1-ubyte" , "rb");
  else
    pFile = fopen("train-labels.idx1-ubyte" , "rb");

  int iteration = 0;

  while (iteration < LABEL_END){

    fread(&read_byte, 1, 1, pFile);

    if (iteration < 4)
      magic_no += (read_byte << (MAX_SHIFT - (iteration*BYTE_SIZE)));
    else if (iteration >=4 && iteration < 8)
      num_images += (read_byte << (MAX_SHIFT - (iteration*BYTE_SIZE)));

    iteration += 1;
  }

  printf("magic_no = %d\n", magic_no);
  printf("num_images = %d\n", num_images);
  int image_no;
  int total_images;

  if (dev)
    total_images = DEV_NUM_IMAGES;
  else
    total_images = TRAIN_NUM_IMAGES;

  for (image_no = 0; image_no < total_images; image_no++){
      fread(&read_byte, 1, 1, pFile);

      if (dev)
	dev_labels[image_no] = read_byte;
      else
	train_labels[image_no] = read_byte;
      iteration += 1;
  }

  fclose (pFile);

  if (dev)
    printf(RED_TEXT("Done parsing development labels\n"));
  else
    printf(RED_TEXT("Done parsing training labels\n"));

  printf("\n");
  return 0;
}


int parse_images(int dev){

  if (dev)
    printf(RED_TEXT("Parsing development images\n"));
  else
    printf(RED_TEXT("Parsing training images\n"));

  FILE * pFile;
  uint8_t read_byte;

  int magic_no   = 0;
  int num_images = 0;
  int num_rows   = 0;
  int num_cols   = 0;

  if (dev)
    pFile = fopen("t10k-images.idx3-ubyte" , "rb");
  else
    pFile = fopen("train-images.idx3-ubyte" , "rb");
  
  int iteration = 0;
  
  while (iteration < META_DATA_END){    
    fread(&read_byte, 1, 1, pFile);
    
    if (iteration < 4) 
      magic_no += (read_byte << (MAX_SHIFT - (iteration*BYTE_SIZE)));
    else if (iteration >=4 && iteration < 8)
      num_images += (read_byte << (MAX_SHIFT - (iteration*BYTE_SIZE)));
    else if (iteration >=8 && iteration < 12)
      num_rows += (read_byte << (MAX_SHIFT - (iteration*BYTE_SIZE)));
    else if (iteration >=12 && iteration < 16)
      num_cols += (read_byte << (MAX_SHIFT - (iteration*BYTE_SIZE)));
        
    iteration += 1;
  }

  printf("magic_no = %d\n", magic_no);
  printf("num_images = %d\n", num_images);
  printf("num_rows = %d\n", num_rows);
  printf("num_cols = %d\n", num_cols);

  int image_no;
  int total_images;

  if (dev)
    total_images = DEV_NUM_IMAGES;
  else
    total_images = TRAIN_NUM_IMAGES;

  for (image_no = 0; image_no < total_images; image_no++){  
    iteration = 0;
    while (iteration < IMAGE_SIZE){
      fread(&read_byte, 1, 1, pFile);
      
      if (dev)
	dev_images[image_no][iteration] = read_byte;
      else
	train_images[image_no][iteration] = read_byte;
      
      iteration += 1;
    }
  }
  
  fclose (pFile);
  
  if (dev)
    printf(RED_TEXT("Done parsing development images\n"));
  else
    printf(RED_TEXT("Done parsing training images\n"));

  printf("\n");
  return 0;
}


/*
void print_image(uint8_t img[IMAGE_SIZE]){

  int iteration = 0;
  uint8_t read_byte = 0;

  printf("\n");
  while (iteration < IMAGE_SIZE){

    read_byte = img[iteration];
    
    if ((iteration % 28 == 0) && (iteration != 0))
      printf("\n");
    printf("%*u", 4, read_byte);

    iteration += 1;
  }
  printf("\n");
  printf("\n");
}

void print_layer_1(float img[NUM_NEURONS]){

  int iteration = 0;
  float read_byte = 0;

  printf("\n");
  while (iteration < NUM_NEURONS){

    read_byte = img[iteration];

    if ((iteration % 12 == 0) && (iteration != 0))
      printf("\n");

    printf("%+*.*f ",1,3,read_byte);

    iteration += 1;
  }
  printf("\n");
  printf("\n");
}


void print_layer_2(float img[NUM_OUTPUTS]){

  int iteration = 0;
  float read_byte = 0;

  printf("\n");
  while (iteration < NUM_OUTPUTS){

    read_byte = img[iteration];

    printf("%*.*f ",1,3,read_byte);

    iteration += 1;
  }
  printf("\n");
  printf("\n");
}

void print_weights_2(float weights[NUM_NEURONS]){

  int iteration = 0;
  float read_weight;
  printf("\n");

  while (iteration < NUM_NEURONS){

    read_weight = weights[iteration];

    if ((iteration % 12 == 0) && (iteration != 0))
      printf("\n");

    printf("%+5.3f ", 2, read_weight);
    iteration += 1;
  }
  printf("\n");
  printf("\n");
}

void print_weights_1(float weights[IMAGE_SIZE]){
  
  int iteration = 0;
  float read_weight;
  printf("\n");
  
  while (iteration < IMAGE_SIZE){
    
    read_weight = weights[iteration];
    
    if ((iteration % 28 == 0) && (iteration != 0))
      printf("\n");
    
    printf("%+5.2f ", read_weight);
    iteration += 1;
  }
  printf("\n");
  printf("\n");
}

*/
