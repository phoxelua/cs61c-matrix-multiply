#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
		   float* kernel)
{
// the x coordinate of the kernel's center
int kern_cent_X = (KERNX - 1)/2;
// the y coordinate of the kernel's center
int kern_cent_Y = (KERNY - 1)/2;

	
//pad with zeros
int x_padding = (4 - (data_size_X % 4)) % 4; //how much we have to pad the rows to be a multiple of 4
int out_size_X = data_size_X + x_padding;
int out_size_Y = data_size_Y;
float newOut[out_size_X * out_size_Y];

if( data_size_X % 4 == 0){ //divisible by 4 case	
	int new_size_X = data_size_X + 2 + x_padding;
	int new_size_Y = data_size_Y + 2;
	int new_size = new_size_X *new_size_Y;	
	//Copy in into newIn
	float newIn[new_size];
	for (int j=0; j<data_size_Y;j++){
	 	for (int i=0; i<data_size_X;i+=4){
	 		_mm_storeu_ps(newIn + i+1 + (j+1)*new_size_X, _mm_loadu_ps(in + i + j*data_size_X));
	 	}
	}

	//Padds left/right cols with zeros
	float zeros[4] = {0,0,0,0};
	 for (int i=0; i<new_size_X/4*4;){
		 _mm_storeu_ps(newIn + i, _mm_loadu_ps(zeros));
		 _mm_storeu_ps(newIn + i + new_size_X*(new_size_Y-1), _mm_loadu_ps(zeros));
		i+=4;
	 
	 }
	 
	//Pad leftover left/right  with zeros
	 for (int i=new_size_X/4*4; i<new_size_X;i++){
		 //_mm_storeu_ps(newIn + i, _mm_loadu_ps(zeros));
		 newIn[i] = 0;
		 newIn[i + new_size_X*(new_size_Y-1)] = 0;
	 }

	 //Pad newIn top/bottom rows with zeros
	 for (int j=0; j<new_size_Y;j++){
	 newIn[j*new_size_X] = 0;
	 newIn[(new_size_X *(j+1)) - 1] = 0; 
	 }

	float newKern[KERNX*KERNY];
	//newKern = (float*) calloc(9, sizeof(float));
	for(int j=0; j<KERNY; j++){
		for (int i=0; i<KERNX;i++){
			*(newKern + (KERNX-1-i) + (KERNY-1-j)*KERNX) = *(kernel + i + j*KERNX);
		}
	}
	
       // main convolution loop
	for(int y = 1; y < new_size_Y-1; y++){ // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < new_size_X-1; x+=4){ // the x coordinate of the output location we're focusing on
			__m128 out_temp = _mm_load_ps(out + ((x-1)+(y-1)*out_size_X));    
			for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel flipped y coordinate      
				for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel flipped x coordinate
					__m128 kern =  _mm_load1_ps(newKern + ((kern_cent_X+i)+(kern_cent_Y+j)*KERNX));
					__m128 in_temp =  _mm_loadu_ps(newIn + ((x+i) + (y+j)*new_size_X));
					__m128 mult = _mm_mul_ps(kern,in_temp);
					out_temp = _mm_add_ps(out_temp, mult);				
				}
			}
			_mm_storeu_ps(out+(x-1)+(y-1)*out_size_X, out_temp);
		}
	}
return 1;
}
else{ //not divisible by 4 case

		//Copy out into newOut
		for(int j = 0; j < out_size_Y; j++){
			for(int i = 0; i < out_size_X; i+=4){
				_mm_storeu_ps(newOut + i + j * out_size_X, _mm_loadu_ps(out + i + j*data_size_X));
			}
		}	

		//Pad out top/bototm rows with 0
	 	for (int j=0; j<out_size_Y;j++){
			//*(newOut + j*out_size_X) = 0;
			//*(newOut + out_size_X *(j+1) - 1) = 0;
	 		newOut[j*out_size_X] = 0;
	 		newOut[(out_size_X *(j+1)) - 1] = 0; 
	 		for(int i=1;i<=x_padding; i++){
				//*(newOut + (out_size_X * (j+1)) - 1 - i) = 0;	
		 		newOut[(out_size_X *(j+1)) -1-i] = 0; 
		 	}
		}



	//	printf("x_padding %d\n", x_padding);
	int new_size_X = data_size_X + 2 + x_padding;
	int new_size_Y = data_size_Y + 2;
	int new_size = new_size_X *new_size_Y;	
	//printf("%d %d \n", new_size_X, new_size_Y);


	
	//Copy in into newIn
	float newIn[new_size];
	for (int j=0; j<data_size_Y;j++){
	 	for (int i=0; i<data_size_X;i+=4){
	 		_mm_storeu_ps(newIn + i+1 + (j+1)*new_size_X, _mm_loadu_ps(in + i + j*data_size_X));
	 	}
	}

	//Padds left/right cols with zeros
	float zeros[4] = {0,0,0,0};
	 for (int i=0; i<new_size_X/4*4;){
		 _mm_storeu_ps(newIn + i, _mm_loadu_ps(zeros));
		 _mm_storeu_ps(newIn + i + new_size_X*(new_size_Y-1), _mm_loadu_ps(zeros));
		 //newIn[i] = 0;
		 //newIn[i + new_size_X*(new_size_Y-1)] = 0;
		i+=4;
	 
	 }
	 
	//Pad leftover left/right  with zeros
	 for (int i=new_size_X/4*4; i<new_size_X;i++){
		 //_mm_storeu_ps(newIn + i, _mm_loadu_ps(zeros));
		 newIn[i] = 0;
		 newIn[i + new_size_X*(new_size_Y-1)] = 0;
	 }

	 //Pad newIn top/bottom rows with zeros
	 for (int j=0; j<new_size_Y;j++){
	 newIn[j*new_size_X] = 0;
	 newIn[(new_size_X *(j+1)) - 1] = 0; 
	 	for(int i=1;i<=x_padding; i++)
		 newIn[(new_size_X *(j+1)) -1-i] = 0; 
	 }




	

	float newKern[KERNX*KERNY];
	//newKern = (float*) calloc(9, sizeof(float));
	for(int j=0; j<KERNY; j++){
		for (int i=0; i<KERNX;i++){
			*(newKern + (KERNX-1-i) + (KERNY-1-j)*KERNX) = *(kernel + i + j*KERNX);
		}
	}
	
       // main convolution loop
	for(int y = 1; y < new_size_Y-1; y++){ // the y coordinate of theoutput location we're focusing on
		for(int x = 1; x < new_size_X-1-x_padding; x+=4){ // the x coordinate of the output location we're focusing on
			__m128 out_temp = _mm_load_ps(newOut + ((x-1)+(y-1)*out_size_X));       
			for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel flipped y coordinate      
				for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel flipped x coordinate
					__m128 kern =  _mm_load1_ps(newKern + ((kern_cent_X+i)+(kern_cent_Y+j)*KERNX));
					__m128 in_temp =  _mm_loadu_ps(newIn + ((x+i) + (y+j)*new_size_X));
					__m128 mult = _mm_mul_ps(kern,in_temp);
					out_temp = _mm_add_ps(out_temp, mult);				
				}
			}
			_mm_storeu_ps(newOut+(x-1)+(y-1)*out_size_X, out_temp);
		}
	}
	
	
	
	
	//Putting the new output matrix back into the original output matrix.

	for(int j = 0; j < data_size_Y; j++){
		for(int i = 0; i < data_size_X/4*4; i+=4){
			_mm_storeu_ps(out + i + j* data_size_X, _mm_loadu_ps(newOut + i + j* out_size_X));
			//out[i + j* data_size_X] = newOut[i + j* out_size_X];
		}
	}

	for(int j = 0; j < data_size_Y; j++){
		for(int i = data_size_X/4*4; i < data_size_X; i++){
			out[i + j* data_size_X] = newOut[i + j* out_size_X];
		}
	}
}

	return 1;
}


