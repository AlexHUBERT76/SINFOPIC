
using Revise
using FITSIO, EasyFITS
using OptimPackNextGen.Powell
using StatsBase,Statistics
using InterpolationKernels




"""
Toy Model (sky to detector version)
"""



function SkyToDet()

    #File handling to change for your system
    directory = "/home/unahzaal/Documents/StageM2/SINFONI_DHTAUB/reduceddatadirty/DH_TAU_B/06-11-2007/Step4/"
    fitsnamefibre = "out_ns_stack_0000.fits"
    pathf = directory*fitsnamefibre
    path = "/home/unahzaal/Documents/StageM2/SINFONI_DHTAUB/rawdata/"


    lamp =read(FITS(path*"SINFO.2007-11-07T11:19:43.995.fits")[1])
    fibre =read(FITS(pathf)[1])

    dark = zeros(Float32,size(lamp)...,3)
    dark[:,:,1] = read(FITS(path*"SINFO.2007-11-07T10:18:16.606.fits")[1])
    dark[:,:,2] = read(FITS(path*"SINFO.2007-11-07T10:23:32.890.fits")[1])
    dark[:,:,3] = read(FITS(path*"SINFO.2007-11-07T10:28:54.896.fits")[1])

    #median filter
    good = median(dark) .-  3*mad(dark) .< mean(dark,dims=3)[205:262,:,1] .< median(dark) .+  3*mad(dark)
    medimg = mapwindow(median, median(dark,dims=3), (5,1,1)) .- median(dark,dims=3) 
    good .*= median(medimg) .-  3*mad(medimg) .< median(medimg,dims=3)[205:262,:,1] .< median(medimg) .+  3*mad(medimg)

    lmp = lamp[205:262,:]
    lmp = convert(Matrix{Float64},lmp)
    fbr = fibre[205:262,:]
    fbr = convert(Matrix{Float64},fbr)


    #ker = CatmullRomSpline(Float64)
    #ker = CatmullRomSpline{Float64}() # there is an issue with different version of InterpolationKernels

    ker = BSpline{3,Float64}()
    indices =  -999:3000
    #lamp cost functions for minimisation
    function pcostl(x)
        p  = projection(Val(:proj),indices, good', lmp', ker,(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j);
        return -sum(p[find_peaks(p; dist=15,nmax=30)])
    end
    #fibre cost functions for minimisation
    function pcostf(x)
        p  = projection(Val(:proj),indices, good, fbr, ker,(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j);
        return -sum(p[32:33])
    end


    x = [  0., 0., 0.];
    #minimisation
    (~,xl) = newuoa(pcostl, x, 1.,  1e-6; verbose = 2, maxeval = 500,check=false)
    (~,xf) = newuoa(pcostf, x, 1.,  1e-6; verbose = 2, maxeval = 500,check=false)

    #reconstruction polynomials
    fcoord(i,j) =  i +  xl[1] + xl[2]*1e-3*j + xl[3]*1e-6*j*j
    fcoordf(i,j) =  i +  xf[1] + xf[2]*1e-3*j + xf[3]*1e-6*j*j

    #projection
    spectre  = projection(Val(:proj), indices,good', lmp', ker,fcoord );
    spectref  = projection(Val(:proj), indices,good, fbr, ker,fcoordf );

    #deprojection
    lampmodel = deprojection(spectre,axes(lmp'), ker,fcoord,indices);
    fibremodel = deprojection(spectref,axes(fbr), ker,fcoordf,indices);

    #deprojection of non deformed image
    lampmodelnonfit = deprojection(spectre,axes(lmp'), ker,(i,j)->i,indices);
    fibremodelnonfit = deprojection(spectref,axes(fbr), ker,(i,j)->i,indices);


    tlampmodel = copy(lampmodel')
    tfibremodel = copy(fibremodel')

    f = FITS("ToyModelImages/lampmodel.fits", "w")
    write(f,lampmodel)
    close(f)
    f = FITS("ToyModelImages/lampmodelnonfit.fits", "w")
    write(f,lampmodelnonfit)
    close(f)
    f = FITS("ToyModelImages/lmp.fits", "w")
    write(f,lmp)
    close(f)
    f = FITS("ToyModelImages/reslmp.fits", "w")
    write(f,Residual(lampmodel,lampmodelnonfit))
    close(f)
    f = FITS("ToyModelImages/restrue.fits", "w")
    write(f,Residual(lmp,tlampmodel))
    close(f)
    f = FITS("ToyModelImages/fibremodel.fits", "w")
    write(f,fibremodel)
    close(f)
    f = FITS("ToyModelImages/fibremodelnonfit.fits", "w")
    write(f,fibremodelnonfit)
    close(f)
    f = FITS("ToyModelImages/fbr.fits", "w")
    write(f,fbr)
    close(f)
    f = FITS("ToyModelImages/resfbr.fits", "w")
    write(f,Residual(fibremodel,fibremodelnonfit))
    close(f)
    f = FITS("ToyModelImages/restruef.fits", "w")
    write(f,Residual(fbr,fibremodel))
    close(f)

end