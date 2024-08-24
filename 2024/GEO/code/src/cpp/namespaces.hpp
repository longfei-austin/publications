#ifndef NAMESPACES_H
#define NAMESPACES_H

#include <map>

#include "mpi.h"

#include "getopt.h"
#include "string.h"

#include "linear_memory.hpp"

#include "helper_mpi.hpp"

// #include <sycl.hpp>

namespace namespace_mpi
{
    inline int rank = -(1<<30);
    inline int size = -(1<<30);
    
    inline MPI_Comm comm_cart = MPI_COMM_NULL;

    inline std::map<char , std::map<char , int> > map_rank
    { { 'x' , std::map<char , int> { {'L' , MPI_PROC_NULL } , {'R' , MPI_PROC_NULL } } } ,
      { 'y' , std::map<char , int> { {'L' , MPI_PROC_NULL } , {'R' , MPI_PROC_NULL } } } };

    inline int & rank_xL = map_rank.at('x').at('L');
    inline int & rank_xR = map_rank.at('x').at('R');

    inline int & rank_yL = map_rank.at('y').at('L');
    inline int & rank_yR = map_rank.at('y').at('R');

    inline int ndims = 2;
    inline int dims[2] = {0,0};
    inline std::map<char , int> map_dim { {'x',-1} , {'y',-1} };


    inline int periods[2] = {true,true};
    inline int reorder = true;

    inline int coords[2] = {-1,-1};
    inline std::map<char , int> map_coord { {'x',-1} , {'y',-1} };
}
namespace ns_mpi = namespace_mpi;


#ifdef SYCLFLAG
namespace namespace_sycl
{
    inline sycl::queue Q {};
    constexpr long unsigned By = 64; 
}
namespace ns_sycl = namespace_sycl;
#endif


namespace namespace_precision
{
#ifndef PRCSFLAG
    using precision = double;
#endif

#if PRCSFLAG == 64
    using precision = double;
#elif PRCSFLAG == 32    
    using precision = float;
#elif PRCSFLAG == 16        
    using precision = _Float16;
#endif

    MPI_Datatype precision_MPI_Datatype = -1;
}
namespace ns_prcs = namespace_precision;


namespace namespace_config
{
    inline std::string medium_name = "homogeneous";

    inline double min_velocity = 1.;

    inline double central_f  = 5.;
    inline double time_delay = 0.25;

    inline int base_ppw = 10;
    inline int ppw      = -1;

    inline long N_wavelength_x = 60;
    inline long N_wavelength_y = 60;

    inline long src_loc_i_base_ppw = 20 * base_ppw;
    inline long src_loc_j_base_ppw = 20 * base_ppw;

    inline long rcv_loc_i_base_ppw = 40 * base_ppw;
    inline long rcv_loc_j_base_ppw = 40 * base_ppw;

    inline int multiple = 1;

    inline double dx = 0./0.;
    inline double dt = 0./0.;

    inline int Nt = 60000;

    inline char char_cpst = '0'; 

    inline int it_debug = -1;

    inline std::map<char , int> map_Nel_entire { {'x', -1} , {'y', -1} };

    inline std::map<char , std::map<char , int> > map_bgn_entire
    { { 'x' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } ,
      { 'y' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } };

    inline std::map<char , std::map<char , int> > map_end_entire 
    { { 'x' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } ,
      { 'y' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } };

    inline std::map<char , int> map_Nel { {'x', -1} , {'y', -1} };

    inline int & Nel_x = map_Nel.at('x');
    inline int & Nel_y = map_Nel.at('y');

    inline std::map<char, std::map<char , std::map<char , int> > > map_len_LIR;

    inline std::map<char , std::map<char , int> > map_len
    { { 'x' , std::map<char , int> { {'N' , 0 } , {'M' , 0 } } } ,
      { 'y' , std::map<char , int> { {'N' , 0 } , {'M' , 0 } } } };

    inline int & Nx = map_len.at('x').at('N');
    inline int & Mx = map_len.at('x').at('M');

    inline int & Ny = map_len.at('y').at('N');
    inline int & My = map_len.at('y').at('M');


    inline std::map<char , std::map<char , int> > map_bgn
    { { 'x' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } ,
      { 'y' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } };

    inline std::map<char , std::map<char , int> > map_end
    { { 'x' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } ,
      { 'y' , std::map<char , int> { {'N' , -1 } , {'M' , -1 } } } };


    inline double rho  = -1.;
    inline double vp   = -1.;
    inline double beta = -1.;


    std::map< std::array<char,2> , std::string > char2mesh
    {
        { {'N','M'} , "Vx" } ,
        { {'M','N'} , "Vy" } ,
        { {'M','M'} , "P"  }
    };

    std::map< std::string , std::array<char,2> > mesh2char
    {
        { "Vx" , {'N','M'} } ,
        { "Vy" , {'M','N'} } ,
        { "P"  , {'M','M'} }
    };
}
namespace ns_config = namespace_config;


namespace namespace_fields
{
    memory_resource<ns_prcs::precision,2> prmt_Vx;
    memory_resource<ns_prcs::precision,2> prmt_Vy;
    memory_resource<ns_prcs::precision,2> prmt_P ;

    memory_resource<ns_prcs::precision,2> soln_Vx;
    memory_resource<ns_prcs::precision,2> soln_Vy;
    memory_resource<ns_prcs::precision,2> soln_P ;

    memory_resource<ns_prcs::precision,2> rths_Vx;
    memory_resource<ns_prcs::precision,2> rths_Vy;
    memory_resource<ns_prcs::precision,2> rths_P ;


    std::map<std::string , memory_resource<ns_prcs::precision,2> *> mesh2prmt
    {
        { "Vx" , & prmt_Vx } ,
        { "Vy" , & prmt_Vy } ,
        { "P"  , & prmt_P  }
    }; 

    std::map<std::string , memory_resource<ns_prcs::precision,2> *> mesh2soln
    {
        { "Vx" , & soln_Vx } ,
        { "Vy" , & soln_Vy } ,
        { "P"  , & soln_P  }
    };

    std::map<std::string , memory_resource<ns_prcs::precision,2> *> mesh2rths
    {
        { "Vx" , & rths_Vx } ,
        { "Vy" , & rths_Vy } ,
        { "P"  , & rths_P  }
    };

    std::map<std::string , decltype(mesh2prmt) > field2mesh2resource
    { 
        {"prmt" , mesh2prmt} , 
        {"soln" , mesh2soln} , 
        {"rths" , mesh2rths} 
    };

    
    memory_resource<ns_prcs::precision> R_signal_Vx;
    memory_resource<ns_prcs::precision> R_signal_Vy;
    memory_resource<ns_prcs::precision> R_signal_P ;


    std::map<std::string , memory_resource<ns_prcs::precision> *> mesh2signal
    {
        { "Vx" , & R_signal_Vx } ,
        { "Vy" , & R_signal_Vy } ,
        { "P"  , & R_signal_P  }
    }; 


    memory_resource<double> R_energy_Vx;
    memory_resource<double> R_energy_Vy;
    memory_resource<double> R_energy_P ;

    std::map<std::string , memory_resource<double> *> mesh2energy
    {
        { "Vx" , & R_energy_Vx } ,
        { "Vy" , & R_energy_Vy } ,
        { "P"  , & R_energy_P  }
    }; 


    memory_resource<double> R_source_Vx;
    memory_resource<double> R_source_Vy;
    memory_resource<double> R_source_P ;

    std::map<std::string , memory_resource<double> *> mesh2source
    {
        { "Vx" , & R_source_Vx } ,
        { "Vy" , & R_source_Vy } ,
        { "P"  , & R_source_P  }
    }; 


    memory_resource<double,2> weight_energy_Vx;
    memory_resource<double,2> weight_energy_Vy;
    memory_resource<double,2> weight_energy_P ;

    std::map<std::string , memory_resource<double,2> *> mesh2weight_energy
    {
        { "Vx" , & weight_energy_Vx } ,
        { "Vy" , & weight_energy_Vy } ,
        { "P"  , & weight_energy_P  }
    }; 


    memory_resource<double,2> weight_source_Vx;
    memory_resource<double,2> weight_source_Vy;
    memory_resource<double,2> weight_source_P ;

    std::map<std::string , memory_resource<double,2> *> mesh2weight_source
    {
        { "Vx" , & weight_source_Vx } ,
        { "Vy" , & weight_source_Vy } ,
        { "P"  , & weight_source_P  }
    }; 
}
namespace ns_fields = namespace_fields;


#ifdef SYCLFLAG
namespace namespace_sycl_fields
{
    sycl_memory_resource<ns_prcs::precision,2> prmt_Vx (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision,2> prmt_Vy (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision,2> prmt_P  (ns_sycl::Q);

    sycl_memory_resource<ns_prcs::precision,2> soln_Vx (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision,2> soln_Vy (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision,2> soln_P  (ns_sycl::Q);

    sycl_memory_resource<ns_prcs::precision,2> rths_Vx (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision,2> rths_Vy (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision,2> rths_P  (ns_sycl::Q);


    std::map<std::string , sycl_memory_resource<ns_prcs::precision,2> *> mesh2prmt
    {
        { "Vx" , & prmt_Vx } ,
        { "Vy" , & prmt_Vy } ,
        { "P"  , & prmt_P  }
    }; 

    std::map<std::string , sycl_memory_resource<ns_prcs::precision,2> *> mesh2soln
    {
        { "Vx" , & soln_Vx } ,
        { "Vy" , & soln_Vy } ,
        { "P"  , & soln_P  }
    };

    std::map<std::string , sycl_memory_resource<ns_prcs::precision,2> *> mesh2rths
    {
        { "Vx" , & rths_Vx } ,
        { "Vy" , & rths_Vy } ,
        { "P"  , & rths_P  }
    };

    std::map<std::string , decltype(mesh2prmt) > field2mesh2resource
    { 
        {"prmt" , mesh2prmt} , 
        {"soln" , mesh2soln} , 
        {"rths" , mesh2rths} 
    };


    sycl_memory_resource<ns_prcs::precision> R_signal_Vx (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision> R_signal_Vy (ns_sycl::Q);
    sycl_memory_resource<ns_prcs::precision> R_signal_P  (ns_sycl::Q);

    std::map<std::string , sycl_memory_resource<ns_prcs::precision> *> mesh2signal
    {
        { "Vx" , & R_signal_Vx } ,
        { "Vy" , & R_signal_Vy } ,
        { "P"  , & R_signal_P  }
    }; 


    sycl_memory_resource<double> R_energy_Vx (ns_sycl::Q);
    sycl_memory_resource<double> R_energy_Vy (ns_sycl::Q);
    sycl_memory_resource<double> R_energy_P  (ns_sycl::Q);

    std::map<std::string , sycl_memory_resource<double> *> mesh2energy
    {
        { "Vx" , & R_energy_Vx } ,
        { "Vy" , & R_energy_Vy } ,
        { "P"  , & R_energy_P  }
    }; 


    sycl_memory_resource<double,2> weight_energy_Vx (ns_sycl::Q);
    sycl_memory_resource<double,2> weight_energy_Vy (ns_sycl::Q);
    sycl_memory_resource<double,2> weight_energy_P  (ns_sycl::Q);

    std::map<std::string , sycl_memory_resource<double,2> *> mesh2weight_energy
    {
        { "Vx" , & weight_energy_Vx } ,
        { "Vy" , & weight_energy_Vy } ,
        { "P"  , & weight_energy_P  }
    }; 
}
namespace ns_sycl_fields = namespace_sycl_fields;
#endif


class class_mesh 
{
public:
    std::string name = "0000";

    std::map<char,char> dir2type { {'x' , 'U'} , {'y' , 'U'} };

    char & T_x = dir2type.at('x');
    char & T_y = dir2type.at('y');

    bool bool_src = false;
    bool bool_rcv = false;

    long I_SRC = -1;
    long I_RCV = -1;

    memory_accessor<ns_prcs::precision,2> hst_prmt;
    memory_accessor<ns_prcs::precision,2> hst_soln;
    memory_accessor<ns_prcs::precision,2> hst_rths;

    std::map< std::string , memory_accessor<ns_prcs::precision,2> & > hst_map_fields
    { { "prmt" , hst_prmt } , { "soln" , hst_soln } , { "rths" , hst_rths } };

    memory_accessor<ns_prcs::precision> hst_R_signal;
    memory_accessor<double>             hst_R_energy;


    memory_accessor<ns_prcs::precision,2> dev_prmt;
    memory_accessor<ns_prcs::precision,2> dev_soln;
    memory_accessor<ns_prcs::precision,2> dev_rths;

    std::map< std::string , memory_accessor<ns_prcs::precision,2> & > dev_map_fields
    { { "prmt" , dev_prmt } , { "soln" , dev_soln } , { "rths" , dev_rths } };

    memory_accessor<ns_prcs::precision> dev_R_signal;
    memory_accessor<double>             dev_R_energy;


    std::map<char , std::map<char , MPI_Datatype> > type_send
    { { 'x' , std::map<char , MPI_Datatype> { {'L' , -1 } , {'R' , -1 } } } ,
      { 'y' , std::map<char , MPI_Datatype> { {'L' , -1 } , {'R' , -1 } } } };

    std::map<char , std::map<char , MPI_Datatype> > type_recv
    { { 'x' , std::map<char , MPI_Datatype> { {'L' , -1 } , {'R' , -1 } } } ,
      { 'y' , std::map<char , MPI_Datatype> { {'L' , -1 } , {'R' , -1 } } } };


    std::map<char , std::map<char , int> > send_offset
    { { 'x' , std::map<char , int> { {'L' , -1 } , {'R' , -1 } } } ,
      { 'y' , std::map<char , int> { {'L' , -1 } , {'R' , -1 } } } };

    std::map<char , std::map<char , int> > recv_offset
    { { 'x' , std::map<char , int> { {'L' , -1 } , {'R' , -1 } } } ,
      { 'y' , std::map<char , int> { {'L' , -1 } , {'R' , -1 } } } };


    std::map< char , std::map< char , std::array<int,2> > > comm_len
    { { 'x' , std::map<char , std::array<int,2>> { {'L' , {-1,-1} } , {'R' , {-1,-1} } } } ,
      { 'y' , std::map<char , std::array<int,2>> { {'L' , {-1,-1} } , {'R' , {-1,-1} } } } };

    std::map< char , std::map< char , std::array<int,2> > > recv_bgn
    { { 'x' , std::map<char , std::array<int,2>> { {'L' , {-1,-1} } , {'R' , {-1,-1} } } } ,
      { 'y' , std::map<char , std::array<int,2>> { {'L' , {-1,-1} } , {'R' , {-1,-1} } } } };

    std::map< char , std::map< char , std::array<int,2> > > send_bgn
    { { 'x' , std::map<char , std::array<int,2>> { {'L' , {-1,-1} } , {'R' , {-1,-1} } } } ,
      { 'y' , std::map<char , std::array<int,2>> { {'L' , {-1,-1} } , {'R' , {-1,-1} } } } };

#ifdef SYCLFLAG
    std::map< char , std::map< char , sycl::event > > recv_copy_event
    { { 'x' , std::map<char , sycl::event> { {'L' , {} } , {'R' , {} } } } ,
      { 'y' , std::map<char , sycl::event> { {'L' , {} } , {'R' , {} } } } };

    std::map< char , std::map< char , sycl::event > > send_copy_event
    { { 'x' , std::map<char , sycl::event> { {'L' , {} } , {'R' , {} } } } ,
      { 'y' , std::map<char , sycl::event> { {'L' , {} } , {'R' , {} } } } };


    std::map< char , sycl::event > take_drvt_event
    { { 'x' , {} } ,
      { 'y' , {} } };

    sycl::event update_soln_event;

    sycl::event apply_source_event;
    sycl::event record_signal_event;
    sycl::event record_energy_event;
#endif

    class_mesh () {}

    void initialize ( char T_x, char T_y )
    {
        this->T_x = T_x;
        this->T_y = T_y;

        this->name = ns_config::char2mesh.at( {T_x,T_y} ); 

        this->hst_prmt.attach_memory( ns_fields::mesh2prmt.at(this->name)->acc );
        this->hst_soln.attach_memory( ns_fields::mesh2soln.at(this->name)->acc );
        this->hst_rths.attach_memory( ns_fields::mesh2rths.at(this->name)->acc );

        this->hst_R_signal.attach_memory( ns_fields::mesh2signal.at(this->name)->acc );
        this->hst_R_energy.attach_memory( ns_fields::mesh2energy.at(this->name)->acc );

#ifdef SYCLFLAG
        this->dev_prmt.attach_memory( ns_sycl_fields::mesh2prmt.at(this->name)->acc );
        this->dev_soln.attach_memory( ns_sycl_fields::mesh2soln.at(this->name)->acc );
        this->dev_rths.attach_memory( ns_sycl_fields::mesh2rths.at(this->name)->acc );

        this->dev_R_signal.attach_memory( ns_sycl_fields::mesh2signal.at(this->name)->acc );
        this->dev_R_energy.attach_memory( ns_sycl_fields::mesh2energy.at(this->name)->acc );
#endif

        {
            using namespace ns_config;
            class_mpi_printf mpi_printf (MPI_COMM_WORLD);

            int x_bgn = map_bgn_entire.at('x').at(T_x);  int x_end = map_end_entire.at('x').at(T_x);
            int y_bgn = map_bgn_entire.at('y').at(T_y);  int y_end = map_end_entire.at('y').at(T_y);

            long src_loc_i = src_loc_i_base_ppw * multiple;  if ( T_x == 'M' ) { src_loc_i += (multiple / 2); }
            long src_loc_j = src_loc_j_base_ppw * multiple;  if ( T_y == 'M' ) { src_loc_j += (multiple / 2); }

            long rcv_loc_i = rcv_loc_i_base_ppw * multiple;  if ( T_x == 'M' ) { rcv_loc_i += (multiple / 2); }
            long rcv_loc_j = rcv_loc_j_base_ppw * multiple;  if ( T_y == 'M' ) { rcv_loc_j += (multiple / 2); }

            int x_pad = map_bgn.at('x').at( this->T_x );
            int y_pad = map_bgn.at('y').at( this->T_y );

            if ( src_loc_i >= x_bgn && src_loc_i < x_end )
            if ( src_loc_j >= y_bgn && src_loc_j < y_end )
            {
                if ( T_x == 'M' && T_y == 'M' ) { bool_src = true; } 
                if ( bool_src ) 
                {
                    this->I_SRC = this->hst_soln.convert_index ( x_pad + src_loc_i - x_bgn , 
                                                                 y_pad + src_loc_j - y_bgn );
                    printf("src %2s : %ld (%ld %ld) coords : (%d %d) \n", name.c_str() , I_SRC , 
                                                                          x_pad + src_loc_i - x_bgn , 
                                                                          y_pad + src_loc_j - y_bgn ,
                                                                          ns_mpi::coords[0] , ns_mpi::coords[1] );
                }
            }


            if ( rcv_loc_i >= x_bgn && rcv_loc_i < x_end )
            if ( rcv_loc_j >= y_bgn && rcv_loc_j < y_end )
            {
                bool_rcv = true;

                if ( ( T_x == 'N' && rcv_loc_i == x_bgn && ns_mpi::dims[0] != 1 ) 
                  || ( T_y == 'N' && rcv_loc_j == y_bgn && ns_mpi::dims[1] != 1 ) ) { bool_rcv = false; }

                if ( bool_rcv )
                { 
                    this->I_RCV = this->hst_soln.convert_index ( x_pad + rcv_loc_i - x_bgn , 
                                                                 y_pad + rcv_loc_j - y_bgn );
                    printf("rcv %2s : %ld (%ld %ld) coords : (%d %d) \n", name.c_str() , I_RCV , 
                                                                          x_pad + rcv_loc_i - x_bgn ,
                                                                          y_pad + rcv_loc_j - y_bgn ,
                                                                          ns_mpi::coords[0] , ns_mpi::coords[1] );
                }
            }
        }


        for ( char side : {'L','R'} )
        {
            using namespace ns_config;

            int block_count  = map_len_LIR.at('x').at(T_x).at(side);
            int block_length = map_len.at('y').at(T_y);
            int stride = ns_config::map_len.at('y').at(T_y);

            MPI_Type_vector(block_count, block_length, stride, ns_prcs::precision_MPI_Datatype, 
                            &type_recv.at('x').at(side));
            MPI_Type_commit(&type_recv.at('x').at(side));
        }

        for ( char side : {'L','R'} )
        {
            using namespace ns_config;

            int block_count  = map_len.at('x').at(T_x);
            int block_length = map_len_LIR.at('y').at(T_y).at(side);
            int stride = ns_config::map_len.at('y').at(T_y);

            MPI_Type_vector(block_count, block_length, stride, ns_prcs::precision_MPI_Datatype, 
                            &type_recv.at('y').at(side));
            MPI_Type_commit(&type_recv.at('y').at(side));
        }

        type_send.at('x').at('L') = type_recv.at('x').at('R');
        type_send.at('x').at('R') = type_recv.at('x').at('L');

        type_send.at('y').at('L') = type_recv.at('y').at('R');
        type_send.at('y').at('R') = type_recv.at('y').at('L');


        {
            using namespace ns_config;
            send_offset.at('x').at('L') = hst_soln.convert_index( map_end.at('x').at(T_x) - map_Nel.at('x') , 0 );
            send_offset.at('x').at('R') = hst_soln.convert_index(                           map_Nel.at('x') , 0 );

            send_offset.at('y').at('L') = hst_soln.convert_index( 0 , map_end.at('y').at(T_y) - map_Nel.at('y') );
            send_offset.at('y').at('R') = hst_soln.convert_index( 0 ,                           map_Nel.at('y') );
        }


        {
            using namespace ns_config;
            recv_offset.at('x').at('R') = hst_soln.convert_index( map_end.at('x').at(T_x) , 0 );
            recv_offset.at('x').at('L') = hst_soln.convert_index(                       0 , 0 );

            recv_offset.at('y').at('R') = hst_soln.convert_index( 0 , map_end.at('y').at(T_y) );
            recv_offset.at('y').at('L') = hst_soln.convert_index(                       0 , 0 );
        }

        {
            using namespace ns_config;

            recv_bgn.at('x').at('L') = { 0 , 0 };
            recv_bgn.at('x').at('R') = { map_end.at('x').at(T_x) , 0 };

            recv_bgn.at('y').at('L') = { 0 , 0 };
            recv_bgn.at('y').at('R') = { 0 , map_end.at('y').at(T_y) };


            send_bgn.at('x').at('L') = { map_end.at('x').at(T_x) - map_Nel.at('x') , 0 };
            send_bgn.at('x').at('R') = {                           map_Nel.at('x') , 0 };

            send_bgn.at('y').at('L') = { 0 , map_end.at('y').at(T_y) - map_Nel.at('y') };
            send_bgn.at('y').at('R') = { 0 ,                           map_Nel.at('y') };


            comm_len.at('x').at('L') = { map_len_LIR.at('x').at(T_x).at('L') , map_len.at('y').at(T_y) };
            comm_len.at('x').at('R') = { map_len_LIR.at('x').at(T_x).at('R') , map_len.at('y').at(T_y) };

            comm_len.at('y').at('L') = { map_len.at('x').at(T_x) , map_len_LIR.at('y').at(T_y).at('L') };
            comm_len.at('y').at('R') = { map_len.at('x').at(T_x) , map_len_LIR.at('y').at(T_y).at('R') };
        }
    }
};


namespace namespace_meshes
{
    class_mesh mesh_Vx;
    class_mesh mesh_Vy;
    class_mesh mesh_P ;

    std::map<std::string , class_mesh *> name2mesh
    {
        {"Vx" , & mesh_Vx } ,
        {"Vy" , & mesh_Vy } ,
        {"P"  , & mesh_P  }
    };
}
namespace ns_meshes = namespace_meshes;


namespace namespace_helper_char
{
    constexpr char flip_side(char side)
    {
             if ( side == 'L' ) { return 'R'; }
        else if ( side == 'R' ) { return 'L'; }
        else throw std::logic_error("side can only be L or R");
    }

    constexpr char flip_type(char type)
    {
             if ( type == 'M' ) { return 'N'; }
        else if ( type == 'N' ) { return 'M'; }
        else throw std::logic_error("type can only be M or N");
    }
}
namespace ns_h_char = namespace_helper_char;


namespace namespace_action
{
    template<char dir>
    void comm_to_halo ( class_mesh & mesh )
    {
        memory_accessor<ns_prcs::precision,2> & soln = mesh.hst_soln;

        for ( char send_side : {'L','R'} )
        {
            char recv_side = ns_h_char::flip_side(send_side);

            void * sendbuf = soln.ptr + mesh.send_offset.at(dir).at(send_side);
            int sendcount = 1;
            MPI_Datatype sendtype = mesh.type_send.at(dir).at(send_side);
            int dest = ns_mpi::map_rank.at(dir).at(send_side);

            void * recvbuf = soln.ptr + mesh.recv_offset.at(dir).at(recv_side);
            int recvcount = 1;
            MPI_Datatype recvtype = mesh.type_recv.at(dir).at(recv_side);
            int orig = ns_mpi::map_rank.at(dir).at(recv_side);

            MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, 1, 
                         recvbuf, recvcount, recvtype, orig, 1, 
                         ns_mpi::comm_cart, MPI_STATUS_IGNORE);
        }
    }


    template<char dir, char type>
    void take_drvt ( int ix , int iy , 
                     class_mesh & drvt_mesh , class_mesh & soln_mesh )
    {
        static_assert(  dir == 'x' ||  dir == 'y' );
        static_assert( type == 'M' || type == 'N' );

        using namespace ns_meshes;

        constexpr int stencil_length = 4;
        constexpr std::array<double, 4> stencil = {1./24 , -9./8 , 9./8 , -1./24};
        constexpr std::array<int, 4> stencil_shift = type == 'N'  
                                                   ? std::array<int, 4> {-1, 0,1,2} 
                                                   : std::array<int, 4> {-2,-1,0,1};

#ifdef PROMOTION
        std::array<double, 4> stencil_dt_dx;
#else
        std::array<ns_prcs::precision, 4> stencil_dt_dx;
#endif
        for ( int iw = 0; iw<4; iw++ ) { stencil_dt_dx[iw] = stencil[iw] * ns_config::dt 
                                                                         / ns_config::dx; }

        static_assert(stencil_length == stencil.size());
        static_assert(stencil_length == stencil_shift.size());
        assert( type == drvt_mesh.dir2type.at(dir) ); 

#ifdef PROMOTION
        double v_d = 0.;
#else
        ns_prcs::precision v_d = 0.;
#endif
        for ( long iw=0; iw<stencil_length; iw++ )
        {
            int ix_s = dir == 'x' ? ix + stencil_shift[iw] : ix;
            int iy_s = dir == 'y' ? iy + stencil_shift[iw] : iy;

            v_d += soln_mesh.hst_soln.at( ix_s , iy_s ) * stencil_dt_dx[iw];
        }
        long ind = drvt_mesh.hst_rths.convert_index(ix,iy);
        drvt_mesh.hst_rths.at(ind) += v_d * drvt_mesh.hst_prmt.at(ind);
    }


    template<typename T>
    void sum_3op ( T const a , T const b ,
                   T &     s , T &     t )
    {
          s = a + b;
        T z = s - a;
          t = b - z;
    }


    template<typename T>
    void sum_6op ( T const a , T const b ,
                   T &     s , T &     t )
    {
            s =   a + b;
        T p_a =   s - b;
        T p_b =   s - p_a;

        T d_a =   a - p_a;
        T d_b =   b - p_b;

            t = d_a + d_b;
    } 


    inline void update_soln ( int ix , int iy , class_mesh & mesh )
    {
        char cpst = ns_config::char_cpst;

        memory_accessor<ns_prcs::precision,2> & soln = mesh.hst_soln;
        memory_accessor<ns_prcs::precision,2> & rths = mesh.hst_rths;        

        long ind = soln.convert_index(ix,iy);
        if ( cpst == '0' ) { soln.at(ind) += rths.at(ind); rths.at(ind) = 0.; }
        if ( cpst == '3' ) { sum_3op<ns_prcs::precision> ( soln.at(ind) , rths.at(ind) , soln.at(ind) , rths.at(ind) ); }
        if ( cpst == '6' ) { sum_6op<ns_prcs::precision> ( soln.at(ind) , rths.at(ind) , soln.at(ind) , rths.at(ind) ); }
    }

    inline void cleanup_MPI ()
    {
        for ( auto iter : ns_meshes::name2mesh )
        {
            class_mesh & mesh = * iter.second;

            for ( char  dir : {'x','y'} )
            for ( char side : {'L','R'} )
            {
                MPI_Type_free( &mesh.type_recv.at(dir).at(side) );
                MPI_Type_free( &mesh.type_send.at(dir).at(side) );
            }
        }
    }
}
namespace ns_action = namespace_action;


#ifdef SYCLFLAG
namespace namespace_sycl_action
{

    template<int N_dim , int message_length>
    inline void print_location ( sycl::queue & Q , memory_accessor<ns_prcs::precision,N_dim> & field , 
                          long location , const std::array<char,message_length> message )
    {
        int it_debug = ns_config::it_debug;

        Q.submit([&](sycl::handler &h) 
        {
            sycl::stream out(1024, 256, h);
            h.single_task([=]() 
            { 
                for ( int i = 0; i < message_length; i++ ) 
                    { out << message[i]; }
                out << " " << it_debug << " " 
                    << sycl::setw(32) << sycl::scientific << sycl::setprecision(32) 
                    << field.at(location) << "\n";
            });
        });
        // Q.wait();
    }


    template<char dir>
    void copy_to_halo ( sycl::queue & Q , class_mesh & mesh )
    {
        memory_accessor<ns_prcs::precision,2> & soln = mesh.dev_soln;

        for ( char send_side : {'L','R'} )
        {
            unsigned long len_0 = mesh.comm_len.at(dir).at(send_side).at(0);
            unsigned long len_1 = mesh.comm_len.at(dir).at(send_side).at(1);

            unsigned long send_bgn_0 = mesh.send_bgn.at(dir).at(send_side).at(0);
            unsigned long send_bgn_1 = mesh.send_bgn.at(dir).at(send_side).at(1);

            char recv_side = ns_h_char::flip_side(send_side);

            unsigned long recv_bgn_0 = mesh.recv_bgn.at(dir).at(recv_side).at(0);
            unsigned long recv_bgn_1 = mesh.recv_bgn.at(dir).at(recv_side).at(1);

            mesh.recv_copy_event.at(dir).at(recv_side) = 
            Q.submit([&](sycl::handler &h) 
            {
                h.depends_on( mesh.update_soln_event );
                h.parallel_for(sycl::range{len_0,len_1}, [=](sycl::id<2> sycl_ind)
                {
                    soln.at(recv_bgn_0 + sycl_ind[0] , recv_bgn_1 + sycl_ind[1]) = 
                    soln.at(send_bgn_0 + sycl_ind[0] , send_bgn_1 + sycl_ind[1]) ;
                });
            });
            // Q.wait();
            mesh.send_copy_event.at(dir).at(send_side) = mesh.recv_copy_event.at(dir).at(recv_side);
        }
    }


    template<char dir, char type>
    void take_drvt ( sycl::queue & Q , 
                     class_mesh & drvt_mesh , class_mesh & soln_mesh )
    {
        static_assert(  dir == 'x' ||  dir == 'y' );
        static_assert( type == 'M' || type == 'N' );

        drvt_mesh.take_drvt_event.at(dir) = 
        Q.submit([&](sycl::handler &h) 
        {
            constexpr int stencil_length = 4;
            constexpr std::array<double, 4> stencil = {1./24 , -9./8 , 9./8 , -1./24};
            constexpr std::array<int, 4> stencil_shift = type == 'N'  
                                                       ? std::array<int, 4> {-1, 0,1,2} 
                                                       : std::array<int, 4> {-2,-1,0,1};
#ifdef PROMOTION
            std::array<double, 4> stencil_dt_dx;
#else
            std::array<ns_prcs::precision, 4> stencil_dt_dx;
#endif
            for ( int iw = 0; iw<4; iw++ ) { stencil_dt_dx[iw] = stencil[iw] * ns_config::dt 
                                                                             / ns_config::dx; }

            static_assert(stencil_length == stencil.size());
            static_assert(stencil_length == stencil_shift.size());
            assert( type == drvt_mesh.dir2type.at(dir) );  

            const uint32_t bgn_x = ns_config::map_bgn.at('x').at(drvt_mesh.T_x);
            const uint32_t bgn_y = ns_config::map_bgn.at('y').at(drvt_mesh.T_y);

            const uint32_t Lx = ns_config::map_end.at('x').at(drvt_mesh.T_x) - bgn_x;
            const uint32_t Ly = ns_config::map_end.at('y').at(drvt_mesh.T_y) - bgn_y;

            memory_accessor<ns_prcs::precision,2> & soln = soln_mesh.dev_soln;
            memory_accessor<ns_prcs::precision,2> & rths = drvt_mesh.dev_rths;
            memory_accessor<ns_prcs::precision,2> & prmt = drvt_mesh.dev_prmt;


            uint32_t const stride_x = drvt_mesh.dev_soln.dir_stride.at(0);
            uint32_t const stride_y = drvt_mesh.dev_soln.dir_stride.at(1);            


            if ( dir == 'y' ) { h.depends_on( {drvt_mesh.take_drvt_event.at('x')} ); }
                
            h.depends_on( {soln_mesh.recv_copy_event.at(dir).at('L')} ); 
            h.depends_on( {soln_mesh.recv_copy_event.at(dir).at('R')} ); 

            constexpr uint32_t By = 64;  
            h.parallel_for(sycl::range{Lx,By}, [=](sycl::id<2> sycl_ind) 
            {
                long const ix   = bgn_x + sycl_ind[0];
                long const iy_B = bgn_y + sycl_ind[1];

                long ind = ix * stride_x + iy_B * stride_y;
                for ( long iy = iy_B; iy < bgn_y + Ly; iy += By )
                {
#ifdef PROMOTION
                    double v_d = 0.;
#else
                    ns_prcs::precision v_d = 0.;
#endif
                    for ( long iw=0; iw<stencil_length; iw++ )
                    {
                        int ix_s = dir == 'x' ? ix + stencil_shift[iw] : ix;
                        int iy_s = dir == 'y' ? iy + stencil_shift[iw] : iy;

                        v_d += soln.at( ix_s , iy_s ) * stencil_dt_dx[iw];
                    }
                    rths.at(ind) += v_d * prmt.at(ind);

                    ind += By * stride_y;
                }
            });
        });
        // Q.wait();
    }


    inline void update_soln ( sycl::queue & Q , class_mesh & mesh )
    {
        mesh.update_soln_event =         
        Q.submit([&](sycl::handler &h) 
        {
            const unsigned long bgn_x = ns_config::map_bgn.at('x').at(mesh.T_x);
            const unsigned long bgn_y = ns_config::map_bgn.at('y').at(mesh.T_y);

            const unsigned long Lx = ns_config::map_end.at('x').at(mesh.T_x) - bgn_x;
            const unsigned long Ly = ns_config::map_end.at('y').at(mesh.T_y) - bgn_y;

            memory_accessor<ns_prcs::precision,2> & soln = mesh.dev_soln;
            memory_accessor<ns_prcs::precision,2> & rths = mesh.dev_rths;

            uint32_t const stride_x = mesh.dev_soln.dir_stride.at(0);
            uint32_t const stride_y = mesh.dev_soln.dir_stride.at(1);

            char cpst = ns_config::char_cpst;

            h.depends_on( {mesh.record_signal_event} ); 
            h.depends_on( {mesh.record_energy_event} ); 
            h.depends_on( {mesh.take_drvt_event.at('x')} );
            h.depends_on( {mesh.take_drvt_event.at('y')} );

            constexpr uint32_t By = 64;  
            h.parallel_for(sycl::range{Lx,By}, [=](sycl::id<2> sycl_ind) 
            {
                long const ix   = bgn_x + sycl_ind[0];
                long const iy_B = bgn_y + sycl_ind[1];

                long ind = ix * stride_x + iy_B * stride_y;
                for ( long iy = iy_B; iy < bgn_y + Ly; iy += By )
                {
                    if ( cpst == '0' ) { soln.at(ind) += rths.at(ind); rths.at(ind) = 0.; }
                    if ( cpst == '3' ) { ns_action::sum_3op<ns_prcs::precision> ( soln.at(ind) , rths.at(ind) , soln.at(ind) , rths.at(ind) ); }
                    if ( cpst == '6' ) { ns_action::sum_6op<ns_prcs::precision> ( soln.at(ind) , rths.at(ind) , soln.at(ind) , rths.at(ind) ); }
                    ind += By * stride_y;                    
                }
            });
        });
        // Q.wait();
    }


    inline void apply_source ( sycl::queue & Q , long it, class_mesh & mesh )
    {
        if ( mesh.bool_src )
        {   
            mesh.apply_source_event =      
            Q.submit([&](sycl::handler &h) 
            {
                memory_accessor<ns_prcs::precision,2> & soln = mesh.dev_soln;
                long I_SRC = mesh.I_SRC;

                using namespace ns_config;

                double A;
                {
                    double t = (it+1-1./2.)*dt - time_delay; 
                    A = (1-2*M_PI*M_PI*central_f*central_f*t*t)*exp(-M_PI*M_PI*central_f*central_f*t*t);
                }

                double dt = ns_config::dt;
                double dx = ns_config::dx;

                h.depends_on( mesh.update_soln_event );
                h.single_task([=]() 
                { 
                    soln.at(I_SRC) += A * dt / (dx * dx); 
                });
            
            });
            // Q.wait();
        }
    }


    inline void record_signal ( sycl::queue & Q , long it, class_mesh & mesh )
    {
        if ( mesh.bool_rcv )
        {
            mesh.record_signal_event = 
            Q.submit([&](sycl::handler &h) 
            {
                memory_accessor<ns_prcs::precision,1> & R_signal = mesh.dev_R_signal;
                memory_accessor<ns_prcs::precision,2> & soln = mesh.dev_soln;
                long I_RCV = mesh.I_RCV;

                h.depends_on( mesh.apply_source_event );
                h.single_task([=]() { R_signal.at(it) = soln.at(I_RCV); });
            });
            // Q.wait();
        }
    }


    inline void record_energy ( sycl::queue & Q , long it, class_mesh & mesh )
    {
        mesh.record_energy_event = 
        Q.submit([&](sycl::handler &h)
        {
            double * ptr_energy = mesh.dev_R_energy.ptr + it;
            auto sumReduction = sycl::reduction(ptr_energy, sycl::plus<>());

            memory_accessor<ns_prcs::precision,2> & soln = mesh.dev_soln;
            memory_accessor<ns_prcs::precision,2> & prmt = mesh.dev_prmt;

            double * weight = ns_sycl_fields::mesh2weight_energy.at(mesh.name)->ptr;

            const unsigned long length = soln.length;

            h.depends_on( mesh.apply_source_event );
            h.parallel_for(sycl::range<1> { length }, sumReduction, 
                             [=](sycl::id<1> sycl_ind, auto& sum) 
            {
                long const ind = sycl_ind[0];
                sum += soln.at(ind) * prmt.at(ind) * weight[ind] * soln.at(ind);
            });
        });
        // Q.wait();
    }


    inline void print_rcv ( sycl::queue & Q , class_mesh & mesh , long I_RCV )
    {
        if ( mesh.bool_rcv )
        {        
            Q.submit([&](sycl::handler &h) 
            {
                sycl::stream out(1024, 256, h);

                memory_accessor<ns_prcs::precision,2> & soln = mesh.dev_soln;
                h.single_task([=]() 
                { 
                    out << sycl::setw(32) << sycl::scientific << sycl::setprecision(32) 
                        << (double) soln.at(I_RCV) << "  ";
                });
            
            });
            // Q.wait();
        }
    }


    inline void print_signal ( sycl::queue & Q , class_mesh & mesh , long it )
    {
        if ( mesh.bool_rcv )
        {
            Q.submit([&](sycl::handler &h) 
            {
                sycl::stream out(1024, 256, h);

                memory_accessor<ns_prcs::precision> & R_signal = mesh.dev_R_signal;
                h.single_task([=]() 
                { 
                    out << sycl::setw(32) << sycl::scientific << sycl::setprecision(32) 
                        << (double) R_signal.at(it) << "  ";
                });
            });
            // Q.wait();
        }
    }


    inline void print_energy ( sycl::queue & Q , class_mesh & mesh , long it )
    {
        Q.submit([&](sycl::handler &h) 
        {
            sycl::stream out(1024, 256, h);

            memory_accessor<double> & R_energy = mesh.dev_R_energy;
            h.single_task([=]() 
            { 
                out << sycl::setw(32) << sycl::scientific << sycl::setprecision(32) 
                    << (double) R_energy.at(it) << "\n";
            });
        });
        // Q.wait();
    }


    template<typename T>
    inline void field_to_hst ( sycl::queue & Q , class_mesh & mesh , std::string field )
    {
        T * hst_ptr = mesh.hst_map_fields.at(field).ptr;
        T * dev_ptr = mesh.dev_map_fields.at(field).ptr;
        long length = mesh.hst_map_fields.at(field).length;

        Q.memcpy(hst_ptr , dev_ptr , length * sizeof(T)).wait();
    }


    template<typename T>
    inline void field_to_dev ( sycl::queue & Q , class_mesh & mesh , std::string field )
    {
        T * hst_ptr = mesh.hst_map_fields.at(field).ptr;
        T * dev_ptr = mesh.dev_map_fields.at(field).ptr;
        long length = mesh.hst_map_fields.at(field).length;

        Q.memcpy(dev_ptr , hst_ptr , length * sizeof(T)).wait();
    }


    template<typename T>
    inline void signal_to_hst ( sycl::queue & Q , class_mesh & mesh )
    {
        T * hst_ptr = mesh.hst_R_signal.ptr;
        T * dev_ptr = mesh.dev_R_signal.ptr;

        Q.memcpy(hst_ptr, dev_ptr, ns_config::Nt * sizeof(T)).wait();
    }


    template<typename T>
    inline void energy_to_hst ( sycl::queue & Q , class_mesh & mesh )
    {
        T * hst_ptr = mesh.hst_R_energy.ptr;
        T * dev_ptr = mesh.dev_R_energy.ptr;

        Q.memcpy(hst_ptr, dev_ptr, ns_config::Nt * sizeof(T)).wait();
    }

}
namespace ns_sycl_action = namespace_sycl_action;
#endif


namespace namespace_inputs
{
    inline void command_line_input_processing( int argc, char* argv[] )
    {
        using namespace ns_config;

        static struct option long_options[] =
        {
            {"medium",    required_argument,  0,  0},
            {"Nt",        required_argument,  0,  0},
            {"multiple",  required_argument,  0,  0},
            {"procs",     required_argument,  0,  0},
            {"cpst",      required_argument,  0,  0},
            {0, 0, 0, 0}
        };

        optind = 1; 
                    
        while (1)
        {
            int option_index = -1;
            int opt = getopt_long_only (argc, argv, "", long_options, &option_index);

            if (opt == 0) 
            {
                if ( strcmp(long_options[option_index].name, "medium"  ) == 0 ) { medium_name = optarg; }

                if ( strcmp(long_options[option_index].name, "Nt"      ) == 0 ) { Nt = int( strtol( optarg , nullptr, 10 ) ); }

                if ( strcmp(long_options[option_index].name, "multiple") == 0 ) { multiple = int( strtol( optarg , nullptr, 10 ) ); }

                if ( strcmp(long_options[option_index].name, "procs"   ) == 0 ) 
                { 
                    std::stringstream ss (optarg);
                    std::string word;
                    int count = 0;
                    while ( ss >> word )
                    {
                        ns_mpi::dims[count] = int( strtol( word.c_str() , nullptr, 10 ) );
                        count++;
                    }
                    assert( count == ns_mpi::ndims && "input needs to match ndims (did you include numbers in quotes)" );
                }

                if ( strcmp(long_options[option_index].name, "cpst"    ) == 0 ) { char_cpst = optarg[0]; }
            }
            if ( opt == -1 ) break;
        }
    }
}
namespace ns_inputs = namespace_inputs;


namespace namespace_energy
{
    inline void prepare_energy_weight ()
    {
        for ( auto mesh : { "Vx" , "Vy" , "P" } )
        {
            char T_x = ns_config::mesh2char.at(mesh).at(0);
            char T_y = ns_config::mesh2char.at(mesh).at(1);

            long len_x = ns_config::map_len.at('x').at(T_x);
            long len_y = ns_config::map_len.at('y').at(T_y);


            auto &      weight =      ns_fields::mesh2weight_energy.at(mesh);
#ifdef SYCLFLAG            
            auto & sycl_weight = ns_sycl_fields::mesh2weight_energy.at(mesh);
#endif
                 weight->allocate_memory({len_x,len_y});
#ifdef SYCLFLAG                 
            sycl_weight->allocate_memory({len_x,len_y});
#endif


            const unsigned long bgn_x = ns_config::map_bgn.at('x').at(T_x);
            const unsigned long bgn_y = ns_config::map_bgn.at('y').at(T_y);

            const unsigned long end_x = ns_config::map_end.at('x').at(T_x);
            const unsigned long end_y = ns_config::map_end.at('y').at(T_y);


            using ns_config::dx;
            weight->acc.set_constant( dx * dx / 2. );

            for ( int ix = 0; ix < len_x; ix++ )
            for ( int iy = 0; iy < len_y; iy++ )
            {
                if ( ix < bgn_x || ix >= end_x || iy < bgn_y || iy >= end_y )
                    { ns_fields::mesh2weight_energy.at(mesh)->acc.at(ix,iy) = 0.; }
            }

            for ( int ix = bgn_x; ix < end_x; ix++ )
            if ( T_y == 'N' )
            {
                ns_fields::mesh2weight_energy.at(mesh)->acc.at(ix,bgn_y  ) /= 2.;
                ns_fields::mesh2weight_energy.at(mesh)->acc.at(ix,end_y-1) /= 2.;
            }

            for ( int iy = bgn_y; iy < end_y; iy++ )
            if ( T_x == 'N' )
            {
                ns_fields::mesh2weight_energy.at(mesh)->acc.at(bgn_x  ,iy) /= 2.;
                ns_fields::mesh2weight_energy.at(mesh)->acc.at(end_x-1,iy) /= 2.;
            }
#ifdef SYCLFLAG
            ns_sycl::Q.memcpy ( sycl_weight->ptr, weight->ptr, weight->length * sizeof(double) ).wait();
#endif
        }
    }

}
namespace ns_energy = namespace_energy;

#endif