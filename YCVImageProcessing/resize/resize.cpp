/*
 * This code came from OpenCV
 * I just had ported OpenCV code as my need
 */

#include "resize.h"

namespace ycv{

const size_t MAX_ESIZE = 16;
const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

//resizeGeneric_<
//    HResizeLinear<uchar, int, short,
//        INTER_RESIZE_COEF_SCALE,
//        HResizeLinearVec_8u32s>,
//    VResizeLinear<uchar, int, short,
//        FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
//        VResizeLinearVec_32s8u> >,e
//0,
//T: uchar
//WT: int
//AT: short

void Hresize(	const unsigned char** src, int** dst, int count,
        const int* xofs, const short* alpha,
        int swidth, int dwidth, int cn, int xmin, int xmax ) const
{
    int dx, k;
    VecOp vecOp;

    int dx0 = vecOp((const unsigned char**)src, (unsigned char**)dst, count,
        xofs, (const unsigned char*)alpha, swidth, dwidth, cn, xmin, xmax );

    for( k = 0; k <= count - 2; k++ )
    {
        const T *S0 = src[k], *S1 = src[k+1];
        WT *D0 = dst[k], *D1 = dst[k+1];
        for( dx = dx0; dx < xmax; dx++ )
        {
            int sx = xofs[dx];
            WT a0 = alpha[dx*2], a1 = alpha[dx*2+1];
            WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
            WT t1 = S1[sx]*a0 + S1[sx + cn]*a1;
            D0[dx] = t0; D1[dx] = t1;
        }

        for( ; dx < dwidth; dx++ )
        {
            int sx = xofs[dx];
            D0[dx] = WT(S0[sx]*ONE); D1[dx] = WT(S1[sx]*ONE);
        }
    }

    for( ; k < count; k++ )
    {
        const T *S = src[k];
        WT *D = dst[k];
        for( dx = 0; dx < xmax; dx++ )
        {
            int sx = xofs[dx];
            D[dx] = S[sx]*alpha[dx*2] + S[sx+cn]*alpha[dx*2+1];
        }

        for( ; dx < dwidth; dx++ )
            D[dx] = WT(S[xofs[dx]]*ONE);
    }
}

inline int Clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}

void ResizeGeneric(	const YMat<unsigned char> &src,
					YMat<unsigned char> &dst,
					const int* xofs, const void* _alpha,
					const int* yofs, const void* _beta,
					int xmin, int xmax, int ksize )
{
    //typedef typename HResize::alpha_type AT;
    //const AT* beta = (const AT*)_beta;
	//Size ssize = src.size(), dsize = dst.size();

	int src_width = src.GetWidth();
	int src_height = src.GetHeight();
	int num_channels = src.GetChannels();
	int dst_width = dst.GetWidth();
	int dst_height = dst.GetHeight();

	int cn = num_channels;
	src_width = src_width * cn;
	dst_width = dst_width * cn;

    xmin = xmin * cn;
    xmax = xmax * cn;
    // image resize is a separable operation. In case of not too strong

//    Range range(0, dsize.height);
//    resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
//        ssize, dsize, ksize, xmin, xmax);
//    parallel_for_(range, invoker, dst.total()/(double)(1<<16));

    const int align_n = 16;
    int bufstep = (dst_width+align_n-1) & (-align_n);
    std::vector<int> buffer;
    buffer.reserve(bufstep*ksize);
    const unsigned char* srows[MAX_ESIZE] = {0, };
    int* rows[MAX_ESIZE] = {0, };
    int prev_sy[MAX_ESIZE] = {0, };

    for(int k=0;k<ksize;k++){
    	prev_sy[k] = -1;
    	rows[k] = (int*)&buffer[0] + bufstep*k;
    }

    short* const alpha = (short* const)_alpha;
    short* const beta = (short* const)_beta;

    for(int dy = 0; dy < dst_height; dy++, beta += ksize ){
        int sy0 = yofs[dy], k0=ksize, k1=0, ksize2 = ksize/2;

        for(int k = 0; k < ksize; k++ ){
            int sy = Clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
            for( k1 = std::max(k1, k); k1 < ksize; k1++ ){
                if( sy == prev_sy[k1] ){ // if the sy-th row has been computed already, reuse it.
                    if( k1 > k ){
                        memcpy( rows[k], rows[k1], bufstep*sizeof(rows[0][0]) );
                    }
                    break;
                }
            }
            if( k1 == ksize ){
                k0 = std::min(k0, k); // remember the first row that needs to be computed
            }
            srows[k] = (unsigned char*)(src.bits() + src_width*sy);
            prev_sy[k] = sy;
        }

        //TODO
        if( k0 < ksize ){
            Hresize( (const unsigned char**)(srows + k0), (int**)(rows + k0), ksize - k0, xofs, (const short*)(alpha),
                    ssize.width, dsize.width, cn, xmin, xmax );
        }
        Vresize( (const int**)rows, (unsigned char*)(dst.data + dst.step*dy), beta, dsize.width );
    }

}

void Resize(const YMat<unsigned char> &src, YMat<unsigned char> &dst, YSize size=YSize(0, 0))
{
	int src_width = src.GetWidth();
	int src_height = src.GetHeight();
	int num_channels = src.GetChannels();

	assert(num_channels==1 && "Only resizing 1 channel images is implemented");

	if(size == YSize(0, 0)){
		dst = YMat<T>(size.GetW(), size.GetH(), num_channels);
	}

	int dst_width = dst.GetWidth();
	int dst_height = dst.GetHeight();

	float inv_scale_x = (float)dst_width/src_width;
	float inv_scale_y = (float)dst_height/src_height;

	const int depth = 0;
	const int cn = 1;
	const float scale_x = 1./inv_scale_x;
	const float scale_y = 1./inv_scale_y;

	int xmin = 0;
	int xmax = dst_width;
	int width = dst_width * cn;

	int ksize = 2;
	int ksize2 = 1;

	std::vector<unsigned char> buffer;
	buffer.reserve((width+dst_height)*(sizeof(int)+sizeof(float)*ksize));


	const bool fixpt = true;
	int* xofs = (int*)(unsigned char*)&buffer[0];
	int* yofs = xofs + width;
	float* alpha = (float*)(yofs + dst_height);
	short* ialpha = (short*)alpha;
	float* beta = alpha + width*ksize;
	short* ibeta = ialpha + width*ksize;
	float cbuf[MAX_ESIZE];

	float fx, fy;
    int sx, sy;

	for(int dx=0;dx<dst_width;dx++){
		fx = (float)((dx+0.5)*scale_x - 0.5);
		sx = std::floor(fx);
		fx = fx - sx;

		if( sx < ksize2-1 ){
			xmin = dx + 1;
			if( sx < 0 ){
				fx = 0;
				sx = 0;
			}
		}
		if( sx + ksize2 >= src_width ){
			xmax = std::min( xmax, dx);
			if( sx >= src_width - 1){
				fx = 0;
				sx = src_width - 1;
			}
		}

		sx = sx*cn;
		for(int k=0; k<cn; k++){
			xofs[dx*cn + k] = sx + k;
		}

		cbuf[0] = 1.f-fx;
		cbuf[1] = fx;

        if( fixpt ){
        	int k;
            for(k = 0; k < ksize; k++ )
                ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
            for( ;k < cn*ksize; k++ )
                ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
        }
        else{
        	int k;
            for(k = 0; k < ksize; k++ ){
                alpha[dx*cn*ksize + k] = cbuf[k];
            }
            for( ;k < cn*ksize; k++ ){
                alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
            }
        }
	}

    for(int dy = 0; dy < dsize.height; dy++ )
    {
		fy = (float)((dy+0.5)*scale_y - 0.5);
		sy = std::floor(fy);
		fy -= sy;

        yofs[dy] = sy;
		cbuf[0] = 1.f - fy;
		cbuf[1] = fy;

        if( fixpt )
        {
            for( k = 0; k < ksize; k++ ){
                ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
            }
        }
        else
        {
            for( k = 0; k < ksize; k++ ){
                beta[dy*ksize + k] = cbuf[k];
            }
        }
    }

}



}
