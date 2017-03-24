#include <limits>
#include <sstream>

#include <ParticleContainer.H>
#include <WarpX_f.H>
#include <WarpX.H>

using namespace amrex;

PhysicalParticleContainer::PhysicalParticleContainer (AmrCore* amr_core, int ispecies,
                                                      const std::string& name)
    : WarpXParticleContainer(amr_core, ispecies),
      species_name(name)
{
    plasma_injector.reset(new PlasmaInjector(species_id, species_name));
    charge = plasma_injector->getCharge();
    mass = plasma_injector->getMass();
}

void PhysicalParticleContainer::InitData() {
    AddParticles(0); // Note - only one level right now
}

void
PhysicalParticleContainer::AllocData ()
{
    // have to resize here, not in the constructor because grids have not
    // been built when constructor was called.
    reserveData();
    resizeData();
}

void
PhysicalParticleContainer::AddNRandomUniformPerCell (int lev, Box part_box) {
    BL_PROFILE("PhysicalParticleContainer::AddNRandomPerCell()");

    charge = plasma_injector->getCharge();
    mass = plasma_injector->getMass();

    const Geometry& geom = Geom(lev);
    const Real* dx  = geom.CellSize();

    if (!part_box.ok()) part_box = geom.Domain();

    Real scale_fac;
    int n_part_per_cell = plasma_injector->numParticlesPerCell();

#if BL_SPACEDIM==3
    scale_fac = dx[0]*dx[1]*dx[2]/n_part_per_cell;
#elif BL_SPACEDIM==2
    scale_fac = dx[0]*dx[1]/n_part_per_cell;
#endif

    std::array<Real,PIdx::nattribs> attribs;
    attribs.fill(0.0);

    // Initialize random generator for normal distribution
    std::default_random_engine generator;
    std::uniform_real_distribution<Real> position_distribution(0.0,1.0);
    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const Box& tile_box = mfi.tilebox();
        const Box& intersectBox = tile_box & part_box;
        if (!intersectBox.ok()) continue;

        RealBox tile_real_box { intersectBox, dx, geom.ProbLo() };

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const auto& boxlo = intersectBox.smallEnd();
        for (IntVect iv = intersectBox.smallEnd(); iv <= intersectBox.bigEnd(); intersectBox.next(iv))
        {
            for (int i_part=0; i_part<n_part_per_cell;i_part++)
            {
                // Randomly generate the positions (uniformly inside each cell)
                Real particle_shift_x = position_distribution(generator);
                Real particle_shift_y = position_distribution(generator);
                Real particle_shift_z = position_distribution(generator);

#if (BL_SPACEDIM == 3)
                Real x = tile_real_box.lo(0) + (iv[0]-boxlo[0] + particle_shift_x)*dx[0];
                Real y = tile_real_box.lo(1) + (iv[1]-boxlo[1] + particle_shift_y)*dx[1];
                Real z = tile_real_box.lo(2) + (iv[2]-boxlo[2] + particle_shift_z)*dx[2];
#elif (BL_SPACEDIM == 2)
                Real x = tile_real_box.lo(0) + (iv[0]-boxlo[0] + particle_shift_x)*dx[0];
                Real y = 0.0;
                Real z = tile_real_box.lo(1) + (iv[1]-boxlo[1] + particle_shift_z)*dx[1];
#endif

                if (plasma_injector->insideBounds(x, y, z)) {
                    Real weight;
                    std::array<Real, 3> u;
                    weight = plasma_injector->getDensity(x, y, z) * scale_fac;
                    plasma_injector->getMomentum(u);
                    attribs[PIdx::w ] = weight;
                    attribs[PIdx::ux] = u[0];
                    attribs[PIdx::uy] = u[1];
                    attribs[PIdx::uz] = u[2];
                    AddOneParticle(lev, grid_id, tile_id, x, y, z, attribs);
                }
            }
        }
    }
}

void
PhysicalParticleContainer::AddNRandomNormal (int lev, Box part_box) {
    BL_PROFILE("PhysicalParticleContainer::AddNRandomNormal()");
    amrex::Abort("PhysicalParticleContainer::AddNRandomNormal() not implemented yet.");
}

void
PhysicalParticleContainer::AddNDiagPerCell (int lev, Box part_box) {
    BL_PROFILE("PhysicalParticleContainer::AddNDiagPerCell()");

    charge = plasma_injector->getCharge();
    mass = plasma_injector->getMass();

    const Geometry& geom = Geom(lev);
    const Real* dx  = geom.CellSize();

    if (!part_box.ok()) part_box = geom.Domain();

    Real scale_fac;
//    int n_part_per_cell_x = plasma_injector->numParticlesPerCellX();
//    int n_part_per_cell_y = plasma_injector->numParticlesPerCellY();
//    int n_part_per_cell_z = plasma_injector->numParticlesPerCellZ();
//    int n_part_per_cell = n_part_per_cell_x * n_part_per_cell_y * n_part_per_cell_z;
    int n_part_per_cell = plasma_injector->numParticlesPerCell();

#if BL_SPACEDIM==3
    scale_fac = dx[0]*dx[1]*dx[2]/n_part_per_cell;
#elif BL_SPACEDIM==2
    scale_fac = dx[0]*dx[1]/n_part_per_cell;
#endif

    std::array<Real,PIdx::nattribs> attribs;
    attribs.fill(0.0);
    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
        const Box& tile_box = mfi.tilebox();
        const Box& intersectBox = tile_box & part_box;
        if (!intersectBox.ok()) continue;

        RealBox tile_real_box { intersectBox, dx, geom.ProbLo() };

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const auto& boxlo = intersectBox.smallEnd();
        for (IntVect iv = intersectBox.smallEnd(); iv <= intersectBox.bigEnd(); intersectBox.next(iv))
        {
            for (int i_part=0; i_part<n_part_per_cell;i_part++)
            {
                Real particle_shift = (0.5+i_part)/n_part_per_cell;
#if (BL_SPACEDIM == 3)
                Real x = tile_real_box.lo(0) + (iv[0]-boxlo[0] + particle_shift)*dx[0];
                Real y = tile_real_box.lo(1) + (iv[1]-boxlo[1] + particle_shift)*dx[1];
                Real z = tile_real_box.lo(2) + (iv[2]-boxlo[2] + particle_shift)*dx[2];
#elif (BL_SPACEDIM == 2)
                Real x = tile_real_box.lo(0) + (iv[0]-boxlo[0] + particle_shift)*dx[0];
                Real y = 0.0;
                Real z = tile_real_box.lo(1) + (iv[1]-boxlo[1] + particle_shift)*dx[1];
#endif

                if (plasma_injector->insideBounds(x, y, z)) {
                    Real weight;
                    std::array<Real, 3> u;
                    weight = plasma_injector->getDensity(x, y, z) * scale_fac;
                    plasma_injector->getMomentum(u);
                    attribs[PIdx::w ] = weight;
                    attribs[PIdx::ux] = u[0];
                    attribs[PIdx::uy] = u[1];
                    attribs[PIdx::uz] = u[2];
                    AddOneParticle(lev, grid_id, tile_id, x, y, z, attribs);
                }
            }
        }
    }
}

void
PhysicalParticleContainer::AddNUniformPerCell (int lev, Box part_box) {
    BL_PROFILE("PhysicalParticleContainer::AddNUniformPerCell()");

    charge = plasma_injector->getCharge();
    mass = plasma_injector->getMass();

    const Geometry& geom = Geom(lev);
    const Real* dx  = geom.CellSize();

    if (!part_box.ok()) part_box = geom.Domain();

    Real scale_fac;
    int n_part_per_cell = plasma_injector->numParticlesPerCell();

#if BL_SPACEDIM==3
    scale_fac = dx[0]*dx[1]*dx[2]/n_part_per_cell;
#elif BL_SPACEDIM==2
    scale_fac = dx[0]*dx[1]/n_part_per_cell;
#endif

    std::array<Real,PIdx::nattribs> attribs;
    attribs.fill(0.0);
    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
        const Box& tile_box = mfi.tilebox();
        const Box& intersectBox = tile_box & part_box;
        if (!intersectBox.ok()) continue;

        RealBox tile_real_box { intersectBox, dx, geom.ProbLo() };

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const auto& boxlo = intersectBox.smallEnd();
        for (IntVect iv = intersectBox.smallEnd(); iv <= intersectBox.bigEnd(); intersectBox.next(iv))
        {
            for (int i_part_x=0; i_part_x<n_part_per_cell;i_part_x++)
            {
              Real particle_shift_x = (0.5+i_part_x)/n_part_per_cell;
              for (int i_part_y=0; i_part_y<n_part_per_cell;i_part_y++)
              {
                Real particle_shift_y = (0.5+i_part_y)/n_part_per_cell;
                for (int i_part_z=0; i_part_z<n_part_per_cell;i_part_z++)
                {
                  Real particle_shift_z = (0.5+i_part_z)/n_part_per_cell;
#if (BL_SPACEDIM == 3)
                  Real x = tile_real_box.lo(0) + (iv[0]-boxlo[0] + particle_shift_x)*dx[0];
                  Real y = tile_real_box.lo(1) + (iv[1]-boxlo[1] + particle_shift_y)*dx[1];
                  Real z = tile_real_box.lo(2) + (iv[2]-boxlo[2] + particle_shift_z)*dx[2];
#elif (BL_SPACEDIM == 2)
                  Real x = tile_real_box.lo(0) + (iv[0]-boxlo[0] + particle_shift_x)*dx[0];
                  Real y = 0.0;
                  Real z = tile_real_box.lo(1) + (iv[1]-boxlo[1] + particle_shift_z)*dx[1];
#endif

                  if (plasma_injector->insideBounds(x, y, z)) {
                      Real weight;
                      std::array<Real, 3> u;
                      weight = plasma_injector->getDensity(x, y, z) * scale_fac;
                      plasma_injector->getMomentum(u);
                      attribs[PIdx::w ] = weight;
                      attribs[PIdx::ux] = u[0];
                      attribs[PIdx::uy] = u[1];
                      attribs[PIdx::uz] = u[2];
                      AddOneParticle(lev, grid_id, tile_id, x, y, z, attribs);
                  }
                }
              }
            }
        }
    }
}

void
PhysicalParticleContainer::FieldGather (int lev,
                                        const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
                                        const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz)
{
    const Geometry& gm  = Geom(lev);

#if (BL_SPACEDIM == 3)
    const Real* dx = gm.CellSize();
#elif (BL_SPACEDIM == 2)
    Real dx[3] = { gm.CellSize(0), std::numeric_limits<Real>::quiet_NaN(), gm.CellSize(1) };
#endif

#if (BL_SPACEDIM == 3)
    long ngx_eb = Ex.nGrow();
    long ngy_eb = ngx_eb;
    long ngz_eb = ngx_eb;
#elif (BL_SPACEDIM == 2)
    long ngx_eb = Ex.nGrow();
    long ngy_eb = 0;
    long ngz_eb = ngx_eb;
#endif

    BL_ASSERT(OnSameGrids(lev,Ex));

    {
	Array<Real> xp, yp, zp;

	for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
	{
	    const Box& box = pti.validbox();

            auto& attribs = pti.GetAttribs();

            auto&  wp = attribs[PIdx::w];
            auto& Exp = attribs[PIdx::Ex];
            auto& Eyp = attribs[PIdx::Ey];
            auto& Ezp = attribs[PIdx::Ez];
            auto& Bxp = attribs[PIdx::Bx];
            auto& Byp = attribs[PIdx::By];
            auto& Bzp = attribs[PIdx::Bz];

            const long np = pti.numParticles();

	    // Data on the grid
	    const FArrayBox& exfab = Ex[pti];
	    const FArrayBox& eyfab = Ey[pti];
	    const FArrayBox& ezfab = Ez[pti];
	    const FArrayBox& bxfab = Bx[pti];
	    const FArrayBox& byfab = By[pti];
	    const FArrayBox& bzfab = Bz[pti];

	    Exp.assign(np,0.0);
	    Eyp.assign(np,0.0);
	    Ezp.assign(np,0.0);
	    Bxp.assign(np,0.0);
	    Byp.assign(np,0.0);
	    Bzp.assign(np,0.0);

	    //
	    // copy data from particle container to temp arrays
	    //
#if (BL_SPACEDIM == 3)
            pti.GetPosition(xp, yp, zp);
#elif (BL_SPACEDIM == 2)
            pti.GetPosition(xp, zp);
            yp.resize(np, std::numeric_limits<Real>::quiet_NaN());
#endif

#if (BL_SPACEDIM == 3)
	    long nx = box.length(0);
	    long ny = box.length(1);
	    long nz = box.length(2);
#elif (BL_SPACEDIM == 2)
	    long nx = box.length(0);
	    long ny = 0;
	    long nz = box.length(1);
#endif
	    RealBox grid_box = RealBox( box, gm.CellSize(), gm.ProbLo() );
#if (BL_SPACEDIM == 3)
	    const Real* xyzmin = grid_box.lo();
#elif (BL_SPACEDIM == 2)
	    Real xyzmin[3] = { grid_box.lo(0), std::numeric_limits<Real>::quiet_NaN(), grid_box.lo(1) };
#endif

	    //
	    // Field Gather
	    //
	    const int ll4symtry          = false;
	    const int l_lower_order_in_v = true;
            long lvect_fieldgathe = 64;
	    warpx_geteb_energy_conserving(&np, xp.data(), yp.data(), zp.data(),
					  Exp.data(),Eyp.data(),Ezp.data(),
					  Bxp.data(),Byp.data(),Bzp.data(),
					  &xyzmin[0], &xyzmin[1], &xyzmin[2],
					  &dx[0], &dx[1], &dx[2],
					  &nx, &ny, &nz, &ngx_eb, &ngy_eb, &ngz_eb,
					  &WarpX::nox, &WarpX::noy, &WarpX::noz,
					  exfab.dataPtr(), eyfab.dataPtr(), ezfab.dataPtr(),
					  bxfab.dataPtr(), byfab.dataPtr(), bzfab.dataPtr(),
					  &ll4symtry, &l_lower_order_in_v,
                                          &lvect_fieldgathe,
		                          &WarpX::field_gathering_algo);
        }
    }
}

void
PhysicalParticleContainer::AddParticles (int lev, Box part_box)
{
    if (plasma_injector->injection_style == "ndiagpercell")
        AddNDiagPerCell(lev, part_box);
    else if (plasma_injector->injection_style == "nrandomuniformpercell")
        AddNRandomUniformPerCell(lev, part_box);
    else if (plasma_injector->injection_style == "nrandomnormal")
        AddNRandomNormal(lev, part_box);
    else if (plasma_injector->injection_style == "nuniformpercell")
        AddNUniformPerCell(lev, part_box);
}

void
PhysicalParticleContainer::Evolve (int lev,
				   const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
				   const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz,
				   MultiFab& jx, MultiFab& jy, MultiFab& jz, Real t, Real dt)
{
    BL_PROFILE("PPC::Evolve()");
    BL_PROFILE_VAR_NS("PPC::Evolve::Copy", blp_copy);
    BL_PROFILE_VAR_NS("PICSAR::FieldGather", blp_pxr_fg);
    BL_PROFILE_VAR_NS("PICSAR::ParticlePush", blp_pxr_pp);
    BL_PROFILE_VAR_NS("PICSAR::CurrentDeposition", blp_pxr_cd);

    const Geometry& gm  = Geom(lev);

#if (BL_SPACEDIM == 3)
    const Real* dx = gm.CellSize();
#elif (BL_SPACEDIM == 2)
    Real dx[3] = { gm.CellSize(0), std::numeric_limits<Real>::quiet_NaN(), gm.CellSize(1) };
#endif

#if (BL_SPACEDIM == 3)
    long ngx_eb = Ex.nGrow();
    long ngy_eb = ngx_eb;
    long ngz_eb = ngx_eb;
    long ngx_j  = jx.nGrow();
    long ngy_j  = ngx_j;
    long ngz_j  = ngx_j;
#elif (BL_SPACEDIM == 2)
    long ngx_eb = Ex.nGrow();
    long ngy_eb = 0;
    long ngz_eb = ngx_eb;
    long ngx_j  = jx.nGrow();;
    long ngy_j  = 0;
    long ngz_j  = ngx_j;
#endif

    BL_ASSERT(OnSameGrids(lev,Ex));

    {
	Array<Real> xp, yp, zp, giv;

	for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
	{
	    const Box& box = pti.validbox();

            auto& attribs = pti.GetAttribs();

            auto&  wp = attribs[PIdx::w];
            auto& uxp = attribs[PIdx::ux];
            auto& uyp = attribs[PIdx::uy];
            auto& uzp = attribs[PIdx::uz];
            auto& Exp = attribs[PIdx::Ex];
            auto& Eyp = attribs[PIdx::Ey];
            auto& Ezp = attribs[PIdx::Ez];
            auto& Bxp = attribs[PIdx::Bx];
            auto& Byp = attribs[PIdx::By];
            auto& Bzp = attribs[PIdx::Bz];

            const long np = pti.numParticles();

	    // Data on the grid
	    const FArrayBox& exfab = Ex[pti];
	    const FArrayBox& eyfab = Ey[pti];
	    const FArrayBox& ezfab = Ez[pti];
	    const FArrayBox& bxfab = Bx[pti];
	    const FArrayBox& byfab = By[pti];
	    const FArrayBox& bzfab = Bz[pti];
	    FArrayBox&       jxfab = jx[pti];
	    FArrayBox&       jyfab = jy[pti];
	    FArrayBox&       jzfab = jz[pti];

	    Exp.assign(np,0.0);
	    Eyp.assign(np,0.0);
	    Ezp.assign(np,0.0);
	    Bxp.assign(np,0.0);
	    Byp.assign(np,0.0);
	    Bzp.assign(np,0.0);

	    giv.resize(np);

	    //
	    // copy data from particle container to temp arrays
	    //
	    BL_PROFILE_VAR_START(blp_copy);
#if (BL_SPACEDIM == 3)
            pti.GetPosition(xp, yp, zp);
#elif (BL_SPACEDIM == 2)
            pti.GetPosition(xp, zp);
            yp.resize(np, std::numeric_limits<Real>::quiet_NaN());
#endif
	    BL_PROFILE_VAR_STOP(blp_copy);

#if (BL_SPACEDIM == 3)
	    long nx = box.length(0);
	    long ny = box.length(1);
	    long nz = box.length(2);
#elif (BL_SPACEDIM == 2)
	    long nx = box.length(0);
	    long ny = 0;
	    long nz = box.length(1);
#endif
	    RealBox grid_box = RealBox( box, gm.CellSize(), gm.ProbLo() );
#if (BL_SPACEDIM == 3)
	    const Real* xyzmin = grid_box.lo();
#elif (BL_SPACEDIM == 2)
	    Real xyzmin[3] = { grid_box.lo(0), std::numeric_limits<Real>::quiet_NaN(), grid_box.lo(1) };
#endif

	    //
	    // Field Gather
	    //
	    const int ll4symtry          = false;
	    const int l_lower_order_in_v = true;
            long lvect_fieldgathe = 64;
	    BL_PROFILE_VAR_START(blp_pxr_fg);
	    warpx_geteb_energy_conserving(&np, xp.data(), yp.data(), zp.data(),
					  Exp.data(),Eyp.data(),Ezp.data(),
					  Bxp.data(),Byp.data(),Bzp.data(),
					  &xyzmin[0], &xyzmin[1], &xyzmin[2],
					  &dx[0], &dx[1], &dx[2],
					  &nx, &ny, &nz, &ngx_eb, &ngy_eb, &ngz_eb,
					  &WarpX::nox, &WarpX::noy, &WarpX::noz,
					  exfab.dataPtr(), eyfab.dataPtr(), ezfab.dataPtr(),
					  bxfab.dataPtr(), byfab.dataPtr(), bzfab.dataPtr(),
					  &ll4symtry, &l_lower_order_in_v,
                                          &lvect_fieldgathe,
		                          &WarpX::field_gathering_algo);
	    BL_PROFILE_VAR_STOP(blp_pxr_fg);

	    //
	    // Particle Push
	    //
	    BL_PROFILE_VAR_START(blp_pxr_pp);
	    warpx_particle_pusher(&np, xp.data(), yp.data(), zp.data(),
				  uxp.data(), uyp.data(), uzp.data(), giv.data(),
				  Exp.dataPtr(), Eyp.dataPtr(), Ezp.dataPtr(),
				  Bxp.dataPtr(), Byp.dataPtr(), Bzp.dataPtr(),
				  &this->charge, &this->mass, &dt,
                                  &WarpX::particle_pusher_algo);
	    BL_PROFILE_VAR_STOP(blp_pxr_pp);

	    //
	    // Current Deposition
	    // xxxxx this part needs to be thread safe if we have OpenMP over tiles
	    //
	    long lvect = 8;
	    BL_PROFILE_VAR_START(blp_pxr_cd);
	    warpx_current_deposition(jxfab.dataPtr(), jyfab.dataPtr(), jzfab.dataPtr(),
				     &np, xp.data(), yp.data(), zp.data(),
				     uxp.data(), uyp.data(), uzp.data(),
				     giv.data(), wp.data(), &this->charge,
				     &xyzmin[0], &xyzmin[1], &xyzmin[2],
				     &dt, &dx[0], &dx[1], &dx[2], &nx, &ny, &nz,
				     &ngx_j, &ngy_j, &ngz_j,
                                     &WarpX::nox,&WarpX::noy,&WarpX::noz,
				     &lvect,&WarpX::current_deposition_algo);
	    BL_PROFILE_VAR_STOP(blp_pxr_cd);

	    //
	    // copy particle data back
	    //
	    BL_PROFILE_VAR_START(blp_copy);
#if (BL_SPACEDIM == 3)
            pti.SetPosition(xp, yp, zp);
#elif (BL_SPACEDIM == 2)
            pti.SetPosition(xp, zp);
#endif
            BL_PROFILE_VAR_STOP(blp_copy);
	}
    }
}
