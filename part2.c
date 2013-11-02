#include <emmintrin.h>
#include <omp.h>
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
int new_size_X = data_size_X + 2 + x_padding;
int new_size_Y = data_size_Y + 2;
int new_size = new_size_X *new_size_Y;	
//printf("%d %d \n", new_size_X, new_size_Y);

if( data_size_X % 4== 0){ //divisible by 4 case	
    



     float newIn[new_size];
     int i, j, k,l;
     //int a,b;
    int blocksize = 100; //16,32, 64(38-39)
	//29-36 by block 100, omp, l+4, sse
    __m128 temp2;
    float* temp1;
    #pragma omp parallel for private(i,j,k,l, temp1, temp2) num_threads(8)
     for (j=0; j<data_size_Y; j+=blocksize){ 
     	for (i=0; i<data_size_X; i+=blocksize){ 
         for(l=j; l<j+blocksize && l<data_size_Y; l++){
            for(k=i; k<i+blocksize && k<data_size_X; k+=4){
			temp1 = newIn + (k+1)+(l+1)*new_size_X;
			temp2 =  _mm_loadu_ps(in + k + l*data_size_X);
			_mm_storeu_ps(temp1, temp2);
			//_mm_storeu_ps(newIn + (k+1+4)+(l+1)*new_size_X, _mm_loadu_ps(in + k+4 + l*data_size_X));
                    //newIn[(k+1)+(l+1)*new_size_X] = in[k+l*data_size_X];
     		  	
		}
	     }
	}
	
     }	

     int a,b;
     if (data_size_X % blocksize != 0){
	//printf("remainder loop X\n");
	//printf("%d\n",data_size_X - data_size_X%blocksize); //seg fault when x=mod is 1-3
	#pragma omp parallel for private(a,b) num_threads(8)
       for(b = 0; b < data_size_Y; b++){
      		for(a = data_size_X - data_size_X%blocksize ; a < data_size_X; a+=4){ 
			_mm_storeu_ps(newIn + (a+1)+(b+1)*new_size_X, _mm_loadu_ps(in + a + b*data_size_X));
            		//newIn[(a+1) + (b+1)*new_size_X] = in[a + b*data_size_X];
	  	}
	}
     }
     if (data_size_Y % blocksize != 0){
	//printf("remainder loop Y\n");
	#pragma omp parallel for private(a,b) num_threads(8)
        for(b = data_size_Y - data_size_Y%blocksize ; b < data_size_Y; b++){
      		for(a = 0; a < data_size_X; a+=4){
			_mm_storeu_ps(newIn + (a+1)+(b+1)*new_size_X, _mm_loadu_ps(in + a + b*data_size_X));
            		//newIn[(a+1) + (b+1)*new_size_X] = in[a + b*data_size_X]; //no difference i think
	  	}
	}
     }


	//Pad left/right rows with zeros
	#pragma omp parallel for private(i) num_threads(8)
	for (int i=0; i<new_size_X;i++){	
		newIn[i] = 0;
		newIn[i + new_size_X*(new_size_Y-1)] = 0;
	}


	//Pad top/bottom columns zeros
	#pragma omp parallel for private(i) num_threads(8)
	for (int i=0; i<new_size_Y;i+=2){
		newIn[i*new_size_X] = 0;
		newIn[(i+1)*new_size_X] = 0;
		newIn[(new_size_X *(i+1)) - 1] = 0;
		newIn[(new_size_X *(i+1+1)) - 1] = 0;
	}


		
	__m128 newKern[9];
	newKern[0] = _mm_load1_ps(kernel + 8);
	newKern[1] = _mm_load1_ps(kernel  + 7);
	newKern[2] = _mm_load1_ps(kernel  + 6);
	newKern[3] = _mm_load1_ps(kernel  + 5);
	newKern[4] = _mm_load1_ps(kernel  + 4);
	newKern[5] = _mm_load1_ps(kernel  + 3);
	newKern[6] =_mm_load1_ps(kernel  + 2);
	newKern[7] = _mm_load1_ps(kernel  + 1);
	newKern[8] =  _mm_load1_ps(kernel);


    // main convolution loop 
	int x,y;
	__m128 out_temp, kern0, kern1, kern2, kern3, kern4, kern5, kern6, kern7, kern8, mult0, mult1, mult2, in_temp0, in_temp1, in_temp2; 
	//j = 0;
	//i = 0;
	//
	//blocksize = 32;
	#pragma omp parallel for  private(y,x,out_temp, kern0, kern1, kern2, kern3, kern4, kern5, kern6, kern7, kern8, mult0, mult1, mult2, in_temp0, in_temp1, in_temp2) num_threads(8)
	//for (j=1;j<new_size_Y-1;j+=blocksize){
	//for (i=1;i<new_size_X-1;i+=blocksize){
	//for(y = j; y<j+blocksize && y < new_size_Y-1; y++){ // the y coordinate of theoutput location we're focusing on
	//for(x = i; x<i+blocksize && x < new_size_X-1;x+=4){ // the x coordinate of the output location we're focusing on
	for(y = 1; y < new_size_Y-1; y++){ // the y coordinate of theoutput location we're focusing on
		for(x = 1; x < new_size_X-1;x+=4){ // the x coordinate of the output location we're focusing on

						
						//First
	          				out_temp = _mm_load_ps(out + ((x-1)+(y-1)*data_size_X));

						kern0 =  *(newKern + 0);
						in_temp0 =  _mm_loadu_ps(newIn + ((x-1) + (y-kern_cent_Y)*new_size_X));
						mult0 = _mm_mul_ps(in_temp0,kern0);
						out_temp = _mm_add_ps(out_temp, mult0);


						kern1 = *(newKern + 1);
						in_temp1 =  _mm_loadu_ps(newIn + ((x-1+1) + (y-kern_cent_Y)*new_size_X));
						mult1 = _mm_mul_ps(in_temp1,kern1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern2 = *(newKern  + 2);
						in_temp2 =  _mm_loadu_ps(newIn + ((x-1+2) + (y-kern_cent_Y)*new_size_X));
						mult2 = _mm_mul_ps(in_temp2, kern2);
						out_temp = _mm_add_ps(out_temp, mult2);


						//Sec
						kern3 = *(newKern + 3);
						in_temp0 = _mm_loadu_ps(newIn + ((x-1) + (y-kern_cent_Y+1)*new_size_X));
						mult0 = _mm_mul_ps(kern3, in_temp0);
						out_temp = _mm_add_ps(out_temp, mult0);

						kern4 = *(newKern  + 4);
						in_temp1 =  _mm_loadu_ps(newIn + ((x-1+1) + (y-kern_cent_Y+1)*new_size_X));
						mult1 = _mm_mul_ps(kern4,in_temp1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern5 =  *(newKern + 5);
						in_temp2 =  _mm_loadu_ps(newIn + ((x-1+2) + (y-kern_cent_Y+1)*new_size_X));
						mult2 = _mm_mul_ps(kern5,in_temp2);
						out_temp = _mm_add_ps(out_temp, mult2);


						//Third
						kern6 =  *(newKern + 6);	
						in_temp0 =  _mm_loadu_ps(newIn + ((x-1) + (y-kern_cent_Y+2)*new_size_X));
						mult0 = _mm_mul_ps(kern6,in_temp0);
						out_temp = _mm_add_ps(out_temp, mult0);


						kern7 =  *(newKern  + 7);	
						in_temp1 =  _mm_loadu_ps(newIn + ((x-1+1) + (y-kern_cent_Y+2)*new_size_X));
						mult1 = _mm_mul_ps(kern7,in_temp1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern8 =  *(newKern + 8);
						in_temp2 =  _mm_loadu_ps(newIn + ((x-1+2) + (y-kern_cent_Y+2)*new_size_X));
						mult2 = _mm_mul_ps(kern8,in_temp2);
						out_temp = _mm_add_ps(out_temp, mult2);

						_mm_storeu_ps(out+(x-1)+(y-1)*data_size_X, out_temp);




		}
	}

	//for(int y = 1; y < new_size_Y-1; y++){ // the y coordinate of theoutput location we're focusing on
	//	for(int x =  (new_size_X-1)/16*16 + 1; x < new_size_X-1; x+=4){ // the x coordinate of the output location we're focusing on
	///		__m128 out_temp = _mm_load_ps(out + ((x-1)+(y-1)*data_size_X));    
	//		for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel flipped y coordinate      
	//			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel flipped x coordinate
	//				__m128 kern =  _mm_load1_ps(newKern + ((kern_cent_X+i)+(kern_cent_Y+j)*KERNX));
	//				__m128 in_temp =  _mm_loadu_ps(newIn + ((x+i) + (y+j)*new_size_X));
	//				__m128 mult = _mm_mul_ps(kern,in_temp);
	//				out_temp = _mm_add_ps(out_temp, mult);				
	//			}
	//		}
	//		_mm_storeu_ps(out+(x-1)+(y-1)*data_size_X, out_temp);
	//	}
	//}

	return 1;
}
else{ //not divisible by 4 case

	//float* newIn = (float*)calloc(new_size, sizeof(float));
	float newIn[new_size];

//COPYING IN TO NEW IN////
     int i, j, k,l;
     int a,b;
     int blocksize = 100;
	//29-36 by block 100, omp, l+4, sse
    #pragma omp parallel for private(i,j,k,l) num_threads(8)
     for (j=0; j<data_size_Y; j+=blocksize){ 
     	for (i=0; i<data_size_X; i+=blocksize){ 
         for(l=j; l<j+blocksize && l<data_size_Y; l++){
            for(k=i; k<i+blocksize && k<data_size_X; k+=4){
		_mm_storeu_ps(newIn + (k+1)+(l+1)*new_size_X, _mm_loadu_ps(in + k + l*data_size_X));
                    //newIn[(k+1)+(l+1)*new_size_X] = in[k+l*data_size_X];
     		  	
		}
	     }
	}
	
     }	

    // int a,b;
     if (data_size_X % blocksize != 0){
	//printf("remainder loop X\n");
	//printf("%d\n",data_size_X - data_size_X%blocksize); //seg fault when x=mod is 1-3
	#pragma omp parallel for private(a,b) num_threads(8)
       for(b = 0; b < data_size_Y; b++){
      		for(a = data_size_X - data_size_X%blocksize; a < data_size_X; a+=4){ 
			_mm_storeu_ps(newIn + (a+1)+(b+1)*new_size_X, _mm_loadu_ps(in + a + b*data_size_X));
            		//newIn[(a+1) + (b+1)*new_size_X] = in[a + b*data_size_X];
	  	}
	}
     }
     if (data_size_Y % blocksize != 0){
	//printf("remainder loop Y\n");
	#pragma omp parallel for private(a,b) num_threads(8)
        for(b = data_size_Y - data_size_Y%blocksize; b < data_size_Y; b++){
      		for(a = 0; a < data_size_X; a+=4){
			_mm_storeu_ps(newIn + (a+1)+(b+1)*new_size_X, _mm_loadu_ps(in + a + b*data_size_X));
            		//newIn[(a+1) + (b+1)*new_size_X] = in[a + b*data_size_X];
	  	}
	}
     }
	/*
	//Copy in into newIn
	float newIn[new_size];
	for (int j=0; j<data_size_Y;j++){
	 	for (int i=0; i<data_size_X;i+=4){
	 		_mm_storeu_ps(newIn + i+1 + (j+1)*new_size_X, _mm_loadu_ps(in + i + j*data_size_X));
	 	}
	}
	*/

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
	#pragma omp parallel for num_threads(8) 
	for (int j=0; j<new_size_Y;j++){
	 newIn[j*new_size_X] = 0;
	 newIn[(new_size_X *(j+1)) - 1] = 0; 
	 	for(int i=1;i<=x_padding; i++)
		 newIn[(new_size_X *(j+1)) -1-i] = 0; 
	 }

	//Flip kernel
	__m128 newKern[9];
	newKern[0] = _mm_load1_ps(kernel + 8);
	newKern[1] = _mm_load1_ps(kernel  + 7);
	newKern[2] = _mm_load1_ps(kernel  + 6);
	newKern[3] = _mm_load1_ps(kernel  + 5);
	newKern[4] = _mm_load1_ps(kernel  + 4);
	newKern[5] = _mm_load1_ps(kernel  + 3);
	newKern[6] =_mm_load1_ps(kernel  + 2);
	newKern[7] = _mm_load1_ps(kernel  + 1);
	newKern[8] =  _mm_load1_ps(kernel);

///MAIN CONVOLUTION LOOP////
// main convolution loop - REMAINDERS
	int x = 0;
	int y = 0;
	__m128 out_temp, kern0, kern1, kern2, kern3, kern4, kern5, kern6, kern7, kern8, mult0, mult1, mult2, in_temp0, in_temp1, in_temp2; 
	//j = 0;
	//i = 0;
	//blocksize = 100;
	//#pragma omp parallel for  private(y,x,out_temp, kern0, kern1, kern2, kern3, kern4, kern5, kern6, kern7, kern8, mult0, mult1, mult2, in_temp0, in_temp1, in_temp2) num_threads(8)
	//for (j=1;j<new_size_Y-1;j+=blocksize){
	//for (i=1;i<new_size_X-1;i+=blocksize){
	//for(y = j; y<j+blocksize && y < new_size_Y-1; y++){ // the y coordinate of theoutput location we're focusing on
	//	for(x = i; x<i+blocksize && x < new_size_X-1;x+=4){ // the x coordinate of the output location we're focusing on
	for(y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
		//printf("ds %d\n", data_size_X/4*4);
		for(x = 0; x < data_size_X/4*4;x+=4){ // the x coordinate of the output location we're focusing on
	          //for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel flipped y coordinate      
                        	//for(i = -kern_cent_X; i <= kern_cent_X; i+=3){ // kernel flipped x coordinate
						//First
						out_temp = _mm_loadu_ps(out+ (x+y*data_size_X));
						//printf("after %d %d\n", x,y);
						kern0 =  *(newKern + 0);
						in_temp0 =  _mm_loadu_ps(newIn + ((x) + (y+1-kern_cent_Y)*new_size_X));
						mult0 = _mm_mul_ps(in_temp0,kern0);
						out_temp = _mm_add_ps(out_temp, mult0);


						kern1 = *(newKern + 1);
						in_temp1 =  _mm_loadu_ps(newIn + ((x+1) + (y+1-kern_cent_Y)*new_size_X));
						mult1 = _mm_mul_ps(in_temp1,kern1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern2 = *(newKern  + 2);
						in_temp2 =  _mm_loadu_ps(newIn + ((x+2) + (y+1-kern_cent_Y)*new_size_X));
						mult2 = _mm_mul_ps(in_temp2, kern2);
						out_temp = _mm_add_ps(out_temp, mult2);


						//Sec
						kern3 = *(newKern + 3);
						in_temp0 = _mm_loadu_ps(newIn + ((x) + (y+1-kern_cent_Y+1)*new_size_X));
						mult0 = _mm_mul_ps(kern3, in_temp0);
						out_temp = _mm_add_ps(out_temp, mult0);

						kern4 = *(newKern  + 4);
						in_temp1 =  _mm_loadu_ps(newIn + ((x+1) + (y+1-kern_cent_Y+1)*new_size_X));
						mult1 = _mm_mul_ps(kern4,in_temp1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern5 =  *(newKern + 5);
						in_temp2 =  _mm_loadu_ps(newIn + ((x+2) + (y+1-kern_cent_Y+1)*new_size_X));
						mult2 = _mm_mul_ps(kern5,in_temp2);
						out_temp = _mm_add_ps(out_temp, mult2);


						//Third
						kern6 =  *(newKern + 6);	
						in_temp0 =  _mm_loadu_ps(newIn + ((x) + (y+1-kern_cent_Y+2)*new_size_X));
						mult0 = _mm_mul_ps(kern6,in_temp0);
						out_temp = _mm_add_ps(out_temp, mult0);


						kern7 =  *(newKern  + 7);	
						in_temp1 =  _mm_loadu_ps(newIn + ((x+1) + (y+1-kern_cent_Y+2)*new_size_X));
						mult1 = _mm_mul_ps(kern7,in_temp1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern8 =  *(newKern + 8);
						in_temp2 =  _mm_loadu_ps(newIn + ((x+2) + (y+1-kern_cent_Y+2)*new_size_X));
						mult2 = _mm_mul_ps(kern8,in_temp2);
						out_temp = _mm_add_ps(out_temp, mult2);

						_mm_storeu_ps(out +(x)+(y)*data_size_X, out_temp);


		}
	}



	
	//printf("data sizes %d %d\n", data_size_X, data_size_Y);
	for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
		for(int x = data_size_X/4*4; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
	//printf("before %d %d %d  %f\n", x, y, x + y*data_size_X, out[x + y*data_size_X]);
	//printf("%d %f \n", x + y*data_size_X, out[x + y*data_size_X]);
			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
						out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * newIn[(x+i+1) + (y+j+1)*new_size_X];
				}
			}
		}
	}


}
	
	return 1;

}
