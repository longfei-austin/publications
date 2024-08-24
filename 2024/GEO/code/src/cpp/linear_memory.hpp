#ifndef LINEAR_MEMORY_H
#define LINEAR_MEMORY_H

#include <iostream>
#include <vector>

#include <cassert>

#include <math.h>
#include <string.h>

#ifdef SYCLFLAG
#include <sycl.hpp>
#endif

template<typename T , int dim=1>
class memory_accessor
{
    public:
        long length = 0;
        T * ptr = nullptr;

        std::array<long,dim> dir_length {{-1}};
        std::array<long,dim> dir_stride {{-1}};        


        void attach_dimension ( std::array<long,dim> const & A_len ) 
        {
            this->dir_length = A_len;

            long deduced_length = 1;
            for ( int i = 0; i < dim; i++ ) { deduced_length *= dir_length.at(i); }
            assert ( length == deduced_length && "Error: length does not match" );

            for ( int i = 0; i < dim; i++ )
            {
                dir_stride.at(i) = 1;
                for ( int j = i + 1; j < dim; j++ ) 
                { 
                    dir_stride.at(i) *= dir_length.at(j); 
                }
            }
        }


    private:
        memory_accessor<T,dim> &
        private_attach_memory (T * p , long const L)
        {
            assert ( ptr == nullptr ); 
            assert ( p   != nullptr );
            assert ( L > 0 );
            assert ( L * sizeof(T) < 1024.*1024.*1024. * 80. 
                     && "requested larger than 80GB; if intended, change the limit in this assert" );

            this->ptr = p;  this->length = L;

            return *this;
        }


    public:
        memory_accessor<T,dim> &
        attach_memory (T * p , std::array<long,dim> const & A_len)
        {
            long L = 1;  for ( unsigned int i = 0; i < A_len.size(); i++ ) { L *= A_len.at(i); }

            private_attach_memory (p , L);
            attach_dimension ( A_len );

            return *this; 
        }


        memory_accessor<T,dim> &
        attach_memory (T * p , long const L)
        { 
            private_attach_memory (p , L);
            attach_dimension ( {L} );

            return *this; 
        }


        memory_accessor<T,dim> &
        attach_memory (memory_accessor<T,dim> const & A)
            { attach_memory (A.ptr , A.dir_length);  return *this; }


        memory_accessor(){}  


        memory_accessor (T * p, std::array<long,dim> const & A_len)
            { attach_memory (p , A_len); }

        memory_accessor (T * p , long const L)
            { attach_memory (p , { L }); }


        // assignment operators
        void operator= ( std::initializer_list<T> il ) 
        { 
            if ( static_cast<long unsigned int>(length) != il.size() ) 
                { printf("Error: length - operator= - memory_accessor.\n"); fflush(stdout); exit(0); }

            long i = 0; for ( auto& e : il ) { this->at(i) = e; i++; }
        }

        void operator*= (T const &a) { for ( long i=0; i<length; i++ ) { ptr[i] *= a; } }
                                                       // { this->operator()(i) *= a; }

        void operator/= (T const &a) { for ( long i=0; i<length; i++ ) { ptr[i] /= a; } }
                                                       // { this->operator()(i) /= a; }

        T* data() { return ptr; }

        T& operator() (long const i) // read and write access
        { 
            assert ( ptr != nullptr && "Error: nullptr - operator() - memory_accessor." );
            assert ( i>=0 && i<length && "Out of bound access - memory_accessor." );

            return ptr[i]; 
        }

        T& at (long const i) // read and write access
        {
            assert ( ptr != nullptr && "Error: nullptr - operator() - memory_accessor." );
            assert ( i>=0 && i<length && "Out of bound access - memory_accessor." );

            return ptr[i];
        }


        T& at (long const i) const // read access
        {
            assert ( ptr != nullptr && "Error: nullptr - operator() - memory_accessor." );
            assert ( i>=0 && i<length && "Out of bound access - memory_accessor." );

            return ptr[i];
        }        


        long convert_index ( std::array<long,dim> const & A_ind ) 
        {
            long ind = 0;
            for ( int i = 0; i < dim; i++ ) { ind += A_ind.at(i) * dir_stride.at(i); }

            for ( int i = 0; i < dim; i++ ) 
                { assert( A_ind.at(i) >= 0 && A_ind.at(i) < dir_length.at(i) ); }
            assert ( ( ind >= 0 && ind < length ) && "Out of bound access - - memory_accessor" );

            return ind;
        }


        long convert_index ( long const & i0, long const & i1 ) const
        {
            static_assert ( dim == 2 , "This function is intended for a 2D array." );

            long ind = i0 * dir_stride[0]
                     + i1 * dir_stride[1];

            return ind;
        }


        long convert_index ( long const & i0, long const & i1, long const & i2 ) 
        {
            static_assert ( dim == 3, "This function is intended for a 3D array." );

            long ind = i0 * dir_stride[0]
                     + i1 * dir_stride[1]
                     + i2 * dir_stride[2];

            return ind;
        }


        T& at (std::array<long,dim> const & A_ind) // read and write access
        {
            assert ( ptr != nullptr && "Error: nullptr - at - memory_accessor" );

            long ind = convert_index (A_ind);            
            return ptr[ind];
        }


        T& at ( long const & i0, long const & i1 ) const // read and write access
        {
            assert ( ptr != nullptr && "Error: nullptr - at - memory_accessor" );

            long ind = convert_index(i0,i1);
            return ptr[ind];
        }

        T& at ( long const & i0, long const & i1, long const & i2 ) // read and write access
        {
            assert ( ptr != nullptr && "Error: nullptr - at - memory_accessor" );

            long ind = convert_index(i0,i1,i2);
            return ptr[ind];
        }


        void set_constant(T const & a) { for ( long i = 0; i < length; i++ ) { ptr[i] = a; } }


        T L2_norm () const
            { return sqrt( L2_norm_square () ); } 

        T L2_norm_square () const
        {
            T norm_square = static_cast<T>(0);
            for ( long i=0; i<length; i++ ) 
                { norm_square += this->ptr[i] * this->ptr[i]; }

            return norm_square;
        }

        void print ()
        {   
            for ( int ix = 0; ix < dir_length.at(0); ix++ )
            {
                for ( int iy = 0; iy < dir_length.at(1); iy++ )
                {
                    printf("%4.0f ", this->at(ix,iy));
                }
                printf("\n");
            }
            printf("\n");
        }

        void print_for_copy ( long const bgn , long const end ) const 
        {   
            printf( "length : %ld \n" , end - bgn );
            printf( "{ { " );
            for ( long i=bgn; i<end; i++ ) 
            { 
                printf("% 16.15e" , static_cast< double > ( this->ptr[i] ) ); 
                if ( i < end-1 ) { printf(" , "); }
            }
            printf( " } };\n\n" );
        }

        T* bgn () { return ptr;          }
        T* end () { return ptr + length; }
}; 
//class memory_accessor


template<typename T , int dim=1>
class memory_resource
{
    public:
        long length = 0;
        T * ptr = nullptr;

        std::string name = "undefined";

        memory_accessor<T,dim> acc;

        memory_resource(){}  

    private:
        void private_allocate (long const L)
        {
            assert ( ptr == nullptr && "ptr is already allocated" );
            assert ( L > 0 );
            assert ( (double) ( L * sizeof(T) ) < 1024.*1024.*1024. * 80.
                     && "requested size larger than 80GB; if intended, change the limit in the assert" );

            constexpr long alignment = 32;    // 32 bytes; 
            long byte_size = ( ( L*sizeof(T) + alignment - 1 ) / alignment ) * alignment;

            ptr = static_cast<T *>( aligned_alloc(alignment, byte_size) );
            assert ( ptr != nullptr && "aligned_alloc returned nullptr" ); 
            memset ( ptr, 0, byte_size );

            this->length = L;
        }        


    public:
        void make_accessor (std::array<long,dim> const & A_len)
            { acc.attach_memory (this->ptr , A_len); }


        void allocate_memory (std::array<long,dim> const & A_len)
        {
            long L = 1;  for ( unsigned int i = 0; i < A_len.size(); i++ ) { L *= A_len.at(i); }
            private_allocate (L);

            make_accessor (A_len);
        }

        void allocate_memory (long const L)
            { private_allocate (L); make_accessor ( {L} ); }


        memory_resource (std::array<long,dim> const & A_len)
            { allocate_memory (A_len); }

        memory_resource (long const L)
            { allocate_memory ( {L} ); }


        void copy ( memory_resource<T> const & v )
        {
            assert( ( this != &v && this->ptr != v.ptr ) && "No self copy" );
            assert( this->ptr != nullptr && v.ptr != nullptr );
            assert( this->length > 0 && this->length == v.length );
            assert ( v.acc.ptr != nullptr && "input accessor not valid" );

            for ( long i=0; i<length; i++ ) { ptr[i] = v.ptr[i]; } 
            this->acc = v.acc;
        }


        // copy ctor
        memory_resource( memory_resource<T> const & v ) : memory_resource ( v.acc.dir_length ) 
            { copy (v); }

        // copy assignment operator
        memory_resource<T> & operator=( memory_resource<T> const & v )
            { copy (v); return *this; }

        // move ctor
        memory_resource( memory_resource<T> && v ) noexcept
        {
            this->length = v.length;  this->ptr = v.ptr;  this->acc = v.acc;

            v.length = 0;             v.ptr = nullptr;
        }

        // move assignment operator
        void operator=( memory_resource<T> && v ) noexcept
        {
            assert( ptr == nullptr && "move assignment is only allowed when moved-to vector is empty" );
            assert( this != &v && this->ptr != v.ptr );

            this->length = v.length;  this->ptr = v.ptr;  this->acc = v.acc;
            v.length = 0;             v.ptr = nullptr;  
        }

        // dtor
        ~memory_resource() { free(ptr); }
}; 
// class memory_resource


#ifdef SYCLFLAG
template<typename T , int dim=1>
class sycl_memory_resource
{
    private:
        sycl::queue * Q;

    public:
        long length = 0;
        T * ptr = nullptr;

        std::string name = "undefined";

        memory_accessor<T,dim> acc;

        sycl_memory_resource(sycl::queue & que) : Q (& que) {}  

    private:
        void private_allocate (long const L)
        {
            assert ( ptr == nullptr && "ptr is already allocated" );
            assert ( L > 0 );
            assert ( (double) ( L * sizeof(T) ) < 1024.*1024.*1024. * 80. 
                     && "requested size larger than 80GB; if intended, change the limit in the assert" );

            constexpr long alignment = 32;    // 32 bytes; 
            long byte_size = ( ( L*sizeof(T) + alignment - 1 ) / alignment ) * alignment;

            ptr = static_cast<T *>( sycl::aligned_alloc_device(alignment, byte_size, *Q) );

            assert ( ptr != nullptr && "aligned_alloc returned nullptr" ); 
            Q->memset ( ptr , 0 , L * sizeof(T) ).wait();

            this->length = L;
        }

    public:
        void make_accessor (std::array<long,dim> const & A_len)
            { acc.attach_memory (this->ptr , A_len); }


        void allocate_memory (std::array<long,dim> const & A_len)
        {
            long L = 1;  for ( unsigned int i = 0; i < A_len.size(); i++ ) { L *= A_len.at(i); }
            private_allocate (L);

            make_accessor (A_len);
        }

        void allocate_memory (long const L)
            { private_allocate (L);  make_accessor ( {L} ); }


        sycl_memory_resource (std::array<long,dim> const & A_len , sycl::queue & que) : Q(& que)
            { allocate_memory (A_len); }

        sycl_memory_resource (long const L , sycl::queue & que) : Q(& que)
            { allocate_memory ( {L} ); }

        void copy ( sycl_memory_resource<T> const & v )
        {
            assert( ( this != &v && this->ptr != v.ptr ) && "No self copy" );
            assert( this->ptr != nullptr && v.ptr != nullptr );
            assert( this->length > 0 && this->length == v.length );
            assert ( v.acc.ptr != nullptr && "input accessor not valid" );

            Q->memcpy( this->ptr , v.ptr , length * sizeof(T) ).wait();
            this->acc = v.acc;
        }

        // copy ctor
        sycl_memory_resource ( sycl_memory_resource<T> const & v ) : sycl_memory_resource ( v.acc.dir_length , * v.Q )
            { copy (v); }

        // copy assignment operator
        void operator=( sycl_memory_resource<T> const & v )
            { Q = v.Q; copy (v); }

        // move ctor
        sycl_memory_resource( sycl_memory_resource<T> && v ) : Q(v.Q)
        {
            this->length = v.length;  this->ptr = v.ptr;  this->acc = v.acc;

            v.length = 0;             v.ptr = nullptr;
        }

        // move assignment operator
        void operator=( sycl_memory_resource<T> && v )
        {
            assert( ptr == nullptr && "move assignment is only allowed when moved-to vector is empty" );
            assert( this != &v && this->ptr != v.ptr );

            this->length = v.length;  this->ptr = v.ptr;  this->acc = v.acc;  Q = v.Q;
            v.length = 0;             v.ptr = nullptr;  
        }

        // dtor
        ~sycl_memory_resource() { sycl::free(ptr,*Q); }
}; 
// class sycl_memory_resource
#endif


#endif