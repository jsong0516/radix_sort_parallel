__kernel void reassemble(
  __global int* in,
  __global int* out,
  __global int* ones,
  __global int* zeros,
  int k,
  int n)
  {
    int idx = get_global_id(0);
    int tid = get_local_id(0);
    __local int buf[128];
    if(idx < n) {
      if((in[idx] >> k) & 0x1)
      {
        buf[tid] = zeros[n - 1] + ones[idx] - 1; 
      }
      else
      {
        buf[tid] = zeros[idx] - 1;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      out[buf[tid]] = in[idx];
    }
  }

__kernel void update(__global int *in,
		     __global int *block,
		     int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  if(idx < n && gid > 0)
    {
      in[idx] = in[idx] + block[gid-1];
    }
}



__kernel void scan(
       __global int *in, 
		   __global int *out, 
		   __global int *bout,
		   __local int *buf, 
		   int v,
		   int k,
		   int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);
  int t, r = 0, w = dim;
  __local int buf2[128];
  
  if(idx<n)
    {
      t = in[idx];
      /* CS194: v==-1 used to signify 
       * a "normal" additive scan
       * used to update the partial scans */
      t = (v==-1) ? t : (v==((t>>k)&0x1)); 
      buf[tid] = t;
    }
  else
    {
      buf[tid] = 0;
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int d = 1; d < dim; d <<= 1) 
  {
    // to avoid data racing
    if (tid >= d) {
       buf2[tid - d] = buf[tid - d];
    }
    // Synchronize
    barrier(CLK_LOCAL_MEM_FENCE);

    // Scan
    if (tid >= d) 
    {
      buf[tid] = buf2[tid - d] + buf[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // Setting back to out
  out[idx] = buf[tid];
  // Set the last element of the buffer to be bout
  if (tid == dim - 1) 
  {
    bout[gid] = buf[tid];
  }
}



