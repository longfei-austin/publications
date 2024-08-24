#include <iostream>
#include <array>
#include <bitset>
#include <filesystem>
#include <algorithm>
#include <string>
#include <numeric>

#include "namespaces.hpp"


double tic,toc;


int main(int argc, char* argv[]) 
{
#ifdef NDEBUG
    printf("NDEBUG is defined.\n");
#endif
// [2023/12/21] NOTE: NDEBUG turns off the asserts.

#ifdef PROMOTION
    printf("PROMOTION is defined.\n");
#endif
// [2023/12/21] NOTE: PROMOTION promotes d_v and stencil_dt_dx to double in take_drvt.

#if PRCSFLAG == 64
    printf("PRCSFLAG is 64 -> f64.\n");
#elif PRCSFLAG == 32    
    printf("PRCSFLAG is 32 -> f32.\n");
#elif PRCSFLAG == 16        
    printf("PRCSFLAG is 16 -> f16.\n");
#endif


    MPI_Init(&argc, &argv);
    using ns_sycl::Q;    

    // Q = sycl::queue( sycl::property::queue::in_order() );
    // std::cout << std::boolalpha << Q.is_in_order () << "\n";
    // // [2023/12/20] NOTE: In order queue for debugging.

    std::cout << "Running on " << Q.get_device().get_info<sycl::info::device::name>() << "\n";


tic = MPI_Wtime();
    {
        using namespace ns_mpi;
        class_mpi_printf mpi_printf (MPI_COMM_WORLD);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);


        MPI_Dims_create(size, ndims, dims);
        ns_inputs::command_line_input_processing(argc, argv);
        // [2023/12/03] NOTE: "command_line_input_processing ()" can be called more than once.

        printf("%d %d\n", ns_config::multiple, ns_config::Nt);

        assert( dims[0] * dims[1] == size );
        if ( rank == 0 ) { printf("dims: %d %d\n", dims[0], dims[1]); }
        
        map_dim.at('x') = dims[0];
        map_dim.at('y') = dims[1];


        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

        MPI_Cart_coords(comm_cart, rank, ndims, coords);
        map_coord.at('x') = coords[0];
        map_coord.at('y') = coords[1];

        mpi_printf("rank: %d ---- coords: (%d %d)\n", rank, coords[0], coords[1]);
        if (rank == 0) { printf("\n"); }

        // retrieve neighbors on x direction
        {
            constexpr int dir  = 0;
            constexpr int disp = 1;
            MPI_Cart_shift(comm_cart, dir, disp, &rank_xL, &rank_xR);
            mpi_printf("rank: %d ---- rank_xL: %d rank_xR: %d\n", rank, rank_xL, rank_xR);
        }

        // retrieve neighbors on y direction
        {
            constexpr int dir  = 1;
            constexpr int disp = 1;
            MPI_Cart_shift(comm_cart, dir, disp, &rank_yL, &rank_yR);
            mpi_printf("rank: %d ---- rank_xL: %d rank_xR: %d\n", rank, rank_yL, rank_yR);
        }
    }


    {
        using namespace ns_prcs;
        if constexpr ( std::is_same_v < precision , double   > ) { precision_MPI_Datatype = MPI_DOUBLE ; }
        if constexpr ( std::is_same_v < precision , float    > ) { precision_MPI_Datatype = MPI_FLOAT  ; }
        if constexpr ( std::is_same_v < precision , _Float16 > ) { precision_MPI_Datatype = MPI_INT16_T; }
    }    


    // ---- "prepare" the values for variables from ns_config
    {
        using namespace ns_config;
        ppw = base_ppw * multiple;


        dx = ( min_velocity / ( central_f * 2.5 ) ) / ppw;
        dt = 1e-4;

        rho  = 1.;
        vp   = 1.;
        beta = 1. / ( rho * vp * vp );


        map_Nel_entire.at('x') = N_wavelength_x * ppw;  // <- Nel_x
        map_Nel_entire.at('y') = N_wavelength_y * ppw;  // <- Nel_y


        for ( const char & dir : { 'x' , 'y' } )
        {
            const int dim_dir  = ns_mpi::map_dim.at(dir);
            const int quotient = map_Nel_entire.at(dir) / dim_dir;
            const int leftover = map_Nel_entire.at(dir) - dim_dir * quotient;

            std::vector<int> Nel_vec ( dim_dir , quotient );
            for ( int i = 0; i < leftover; i++ ) { Nel_vec.at(i) += 1; }

            map_Nel.at(dir) = Nel_vec.at( ns_mpi::map_coord.at(dir) );

            map_bgn_entire.at(dir).at('M') = std::accumulate( Nel_vec.begin(), 
                                                              Nel_vec.begin() + ns_mpi::map_coord.at(dir), 0 );
            map_end_entire.at(dir).at('M') = map_bgn_entire.at(dir).at('M') + map_Nel.at(dir);

            map_bgn_entire.at(dir).at('N') = map_bgn_entire.at(dir).at('M');

            map_end_entire.at(dir).at('N') = map_end_entire.at(dir).at('M') + 1;
        }


        class_mpi_printf mpi_printf (MPI_COMM_WORLD);
        for ( const char type : { 'M' , 'N' } )
        for ( const char dir  : { 'x' , 'y' } )
        {
            mpi_printf( "%c%c coords : (%d %d)  (bgn,end) : (% 4d % 4d) \n", dir, type, 
                        ns_mpi::coords[0], ns_mpi::coords[1], 
                        map_bgn_entire.at(dir).at(type), map_end_entire.at(dir).at(type) );
        }


        for ( char dir : {'x','y'} )
        {
            map_len_LIR[dir]['N']['I'] = map_Nel.at(dir) + 1;
            map_len_LIR[dir]['N']['L'] = 1;
            map_len_LIR[dir]['N']['R'] = 1;

            map_len_LIR[dir]['M']['I'] = map_Nel.at(dir);
            map_len_LIR[dir]['M']['L'] = 2;
            map_len_LIR[dir]['M']['R'] = 2;

            // [2023/11/22]
            // NOTE: This setting has one more grid points on 'N' grid than 
            //       on 'M' grid. Their paddings are "symmetric".

            // map_len_LIR[dir]['N']['I'] = map_Nel.at(dir);
            // map_len_LIR[dir]['N']['L'] = 1;
            // map_len_LIR[dir]['N']['R'] = 2;
            // 
            // map_len_LIR[dir]['M']['I'] = map_Nel.at(dir);
            // map_len_LIR[dir]['M']['L'] = 2;
            // map_len_LIR[dir]['M']['R'] = 1;
            //       
            // [2023/11/22]
            // NOTE: This setting has the same grid points on 'N' and 'M' grids, 
            //       but their paddings are "asymmetric".
        }


        for ( char dir : {'x','y'} )
        for ( char T : {'N' , 'M'} )
        {
            assert ( ( map_len_LIR.at(dir).at(T).at('I') >= map_len_LIR.at(dir).at(T).at('L') 
                    && map_len_LIR.at(dir).at(T).at('I') >= map_len_LIR.at(dir).at(T).at('R') )
                    && "There needs to be enough interior points to fill in neighbors halo.") ;
        }


        for ( char dir : {'x','y'} )
        for ( char T : {'N' , 'M'} )
        {
            map_len.at(dir).at(T) = map_len_LIR.at(dir).at(T).at('L')
                                  + map_len_LIR.at(dir).at(T).at('I')
                                  + map_len_LIR.at(dir).at(T).at('R');

            map_bgn.at(dir).at(T) = 0                     + map_len_LIR.at(dir).at(T).at('L');
            map_end.at(dir).at(T) = map_len.at(dir).at(T) - map_len_LIR.at(dir).at(T).at('R');
        }
    }
toc = MPI_Wtime(); printf("\n setup time : %5.4f\n", toc - tic);


tic = MPI_Wtime();
    {
        using namespace ns_config;

        for ( auto field : { "prmt" , "soln" , "rths" } )
        for ( auto mesh  : { "Vx" , "Vy" , "P" } )
        {
            std::array<char,2> mesh_type = ns_config::mesh2char.at(mesh);
            long len_x = map_len.at('x').at( mesh_type.at(0) );
            long len_y = map_len.at('y').at( mesh_type.at(1) );

                 memory_resource<ns_prcs::precision,2> *      res =      ns_fields::field2mesh2resource.at(field).at(mesh);
            sycl_memory_resource<ns_prcs::precision,2> * sycl_res = ns_sycl_fields::field2mesh2resource.at(field).at(mesh);

                 res->allocate_memory ( {len_x , len_y} );
            sycl_res->allocate_memory ( {len_x , len_y} );
        }


        ns_fields::prmt_Vx.acc.set_constant( 1. / rho  );
        ns_fields::prmt_Vy.acc.set_constant( 1. / rho  );
        ns_fields::prmt_P .acc.set_constant( 1. / beta );
    }

toc = MPI_Wtime(); printf("\n prepare field time: %5.4f\n", toc - tic);


tic = MPI_Wtime();
    {
        for ( auto mesh : { "Vx" , "Vy" , "P" } )
        { 
                 ns_fields::mesh2signal.at(mesh)->allocate_memory (ns_config::Nt); 
            ns_sycl_fields::mesh2signal.at(mesh)->allocate_memory (ns_config::Nt); 

                 ns_fields::mesh2energy.at(mesh)->allocate_memory (ns_config::Nt); 
            ns_sycl_fields::mesh2energy.at(mesh)->allocate_memory (ns_config::Nt);             
        }
    }
toc = MPI_Wtime(); printf("\n allocate signal time : %5.4f\n", toc - tic);


tic = MPI_Wtime();
    {
        ns_energy::prepare_energy_weight ();
    }
toc = MPI_Wtime(); if ( ns_mpi::rank==0 ) { printf("\n prepare energy time : %5.4f\n", toc - tic); }


tic = MPI_Wtime();
    // ---- prepare the Grid class
    {
        using namespace ns_meshes;
        using namespace ns_fields;

        mesh_Vx.initialize( 'N', 'M' );
        mesh_Vy.initialize( 'M', 'N' );
        mesh_P .initialize( 'M', 'M' );

        for ( const auto & iter : name2mesh )
            { assert( iter.first == iter.second->name ); }
    }
toc = MPI_Wtime(); printf("\n prepare mesh time : %5.4f\n", toc - tic);


tic = MPI_Wtime();

    // ---- copy the field data from hst to dev
    {
        for ( auto field : { "prmt" , "soln" , "rths" } )
        for ( auto mesh  : { "Vx" , "Vy" , "P" } )
            { ns_sycl_action::field_to_dev<ns_prcs::precision> (Q, * ns_meshes::name2mesh.at(mesh), field); }
    }

toc = MPI_Wtime(); printf("\n hst to dev field copy time : %5.4f\n", toc - tic);


Q.wait();
tic = MPI_Wtime();

    for (int it=0; it<ns_config::Nt; it++)
    {
        // Q.wait();
        if ( (it % 1000) == 0 ) 
        {   
            printf("\n %6d", it);  fflush(stdout);
            for ( auto name : { "Vx" , "Vy" , "P" } )
            { 
                class_mesh & mesh = * ns_meshes::name2mesh.at(name);
                ns_sycl_action::print_rcv(Q, mesh, mesh.I_RCV); // Q.wait();
            }
            printf("\n");
        }


        // update soln_Vx
        {
            ns_sycl_action::copy_to_halo<'x'> ( Q , ns_meshes::mesh_P );
            ns_sycl_action::take_drvt<'x','N'> ( Q , ns_meshes::mesh_Vx , ns_meshes::mesh_P );
            ns_sycl_action::update_soln ( Q , ns_meshes::mesh_Vx );
        }
        

        // update soln_Vy
        {
            ns_sycl_action::copy_to_halo<'y'> ( Q , ns_meshes::mesh_P );
            ns_sycl_action::take_drvt<'y','N'> ( Q , ns_meshes::mesh_Vy , ns_meshes::mesh_P );
            ns_sycl_action::update_soln ( Q , ns_meshes::mesh_Vy );
        }


        // update soln_P
        {
            ns_sycl_action::copy_to_halo<'x'> ( Q , ns_meshes::mesh_Vx );
            ns_sycl_action::copy_to_halo<'y'> ( Q , ns_meshes::mesh_Vy );

            ns_sycl_action::take_drvt<'x','M'> ( Q , ns_meshes::mesh_P , ns_meshes::mesh_Vx );
            ns_sycl_action::take_drvt<'y','M'> ( Q , ns_meshes::mesh_P , ns_meshes::mesh_Vy );

            ns_sycl_action::update_soln ( Q , ns_meshes::mesh_P );

            ns_sycl_action::apply_source ( Q , it , ns_meshes::mesh_P );
        }


        // record solution
        {
            ns_sycl_action::record_signal ( Q , it , ns_meshes::mesh_Vx );
            ns_sycl_action::record_signal ( Q , it , ns_meshes::mesh_Vy );
            ns_sycl_action::record_signal ( Q , it , ns_meshes::mesh_P  );
        }


        // record energy
        {
            ns_sycl_action::record_energy ( Q , it , ns_meshes::mesh_Vx );
            ns_sycl_action::record_energy ( Q , it , ns_meshes::mesh_Vy );
            ns_sycl_action::record_energy ( Q , it , ns_meshes::mesh_P  );
        }
    } // for (int it=0; it<ns_config::Nt; it++)

Q.wait();
toc = MPI_Wtime(); printf("\nloop time : %5.4f\n", toc - tic);


tic = MPI_Wtime();
    for ( auto mesh  : { "Vx" , "Vy" , "P" } )
    {
        ns_sycl_action::signal_to_hst <ns_prcs::precision> ( Q , * ns_meshes::name2mesh.at(mesh) );
        ns_sycl_action::energy_to_hst <         double   > ( Q , * ns_meshes::name2mesh.at(mesh) );
        // [2024/02/26] NOTE: energy_to_hst should take "double" as the template parameter.
    }
toc = MPI_Wtime(); printf("\n dev to hst record copy time : %5.4f\n", toc - tic);


tic = MPI_Wtime();
    // ---- output the solution and energy
    {
        using namespace ns_config;
        using namespace ns_fields;

        std::string dt_folder;
        {
            std::ostringstream ss; 
            ss << std::scientific << std::setprecision(6) << dt;
            dt_folder = "dt_" + ss.str();
            std::replace( dt_folder.begin(), dt_folder.end(), '.', 'p' );
        }

        std::string str_precision = "Unrecognized type";
        if ( std::is_same_v < ns_prcs::precision , double   > ) { str_precision = "f64"; }
        if ( std::is_same_v < ns_prcs::precision , float    > ) { str_precision = "f32"; }
        if ( std::is_same_v < ns_prcs::precision , _Float16 > ) { str_precision = "f16"; }
        printf( "\nPrecision type: %s .\n", str_precision.c_str() );

        std::string folder_name = "../result/acoustic/" + medium_name 
                                + "/src_" + std::to_string(src_loc_i_base_ppw) 
                                +     "_" + std::to_string(src_loc_j_base_ppw)
                                + "_rcv_" + std::to_string(rcv_loc_i_base_ppw) 
                                +     "_" + std::to_string(rcv_loc_j_base_ppw)
                                + "/base_size" + "_" + std::to_string(N_wavelength_x)
                                               + "_" + std::to_string(N_wavelength_y)
                                + "/base_ppw_" + std::to_string(base_ppw)
                                + "/multiple_" + std::to_string(multiple)
                                + "/Nt_" + std::to_string(Nt)                                
                                + "/" + dt_folder
                                + "/" + str_precision
                                + "/" + "cpst_" + ns_config::char_cpst;

#ifdef PROMOTION
        folder_name += "_promotion";
#endif

        if ( ns_mpi::rank == 0 )
        {
            std::filesystem::create_directories( folder_name );
            std::cout << "Results will be stored in folder: \n" << folder_name << std::endl;
            // [2023/09/20] NOTE: create_directories () can take both char* and std::string.
        }
        MPI_Barrier(MPI_COMM_WORLD);


        for ( const std::string name : {"Vx", "Vy", "P"} )
        {
            if ( ns_meshes::name2mesh.at(name)->bool_rcv )
            {
                std::string file_name = folder_name + "/" + name; 
                FILE * fp = fopen( ( file_name + ".txt" ).c_str(), "w" );
                for ( int it = 0; it < Nt; it++ ) 
                    { fprintf( fp, "% 16.15e\n", (double) ns_meshes::name2mesh.at(name)->hst_R_signal.at(it) ); }
                fclose(fp);
            }
        }


        {
            using namespace ns_config;
            using namespace ns_meshes;
            memory_resource<double> res_E (Nt);
            memory_accessor<double> & E = res_E.acc;
            for ( int it=1; it<Nt; it++ )
            { 
                E(it) = mesh_Vx.hst_R_energy(it)       + mesh_Vy.hst_R_energy(it) 
                      + mesh_P. hst_R_energy (it) / 2. + mesh_P .hst_R_energy (it-1) / 2.; 
            }
            E(0) = mesh_Vx.hst_R_energy(0) + mesh_Vy.hst_R_energy(0) + mesh_P.hst_R_energy(0) / 2. + 0. / 2.;


            if ( ns_mpi::rank ==0 ) { MPI_Reduce(MPI_IN_PLACE,   E.ptr, E.length, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); }
            if ( ns_mpi::rank !=0 ) { MPI_Reduce(       E.ptr, nullptr, E.length, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); }


            if ( ns_mpi::rank ==0 )
            {
                std::string file_name = folder_name + "/E";
                FILE * fp = fopen( ( file_name + ".txt" ).c_str(), "w" );
                for ( int it = 0; it < Nt; it++ )
                    { fprintf( fp, "% 16.15e\n", (double) E(it) ); }
                fclose(fp);
            }
        }

    }
toc = MPI_Wtime(); printf("\n write output time : %5.4f\n", toc - tic);

    ns_action::cleanup_MPI();

    MPI_Finalize();

    return 0;
}