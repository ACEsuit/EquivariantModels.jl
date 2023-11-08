using EquivariantModels, Polynomials4ML

totdeg = 9
ν = 4
L = 0
basis = legendre_basis(totdeg)
radial = EquivariantModels.simple_radial_basis(basis)

Aspec, AAspec = EquivariantModels.degord2spec(radial;totaldegree=totdeg,order=ν,Lmax=L,islong = false, rSH=false)
AAspec
Aspec_real, AAspec_real = EquivariantModels.degord2spec(radial;totaldegree=totdeg,order=ν,Lmax=L,islong = false, rSH=true)
AAspec_real

findall(x -> length(x) == 2, AAspec)
findall(x -> length(x) == 2, AAspec_real)
