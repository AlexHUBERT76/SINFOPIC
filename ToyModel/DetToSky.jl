"""
Toy Model (detector to sky version)
"""
function DetToSky()

    timestart = time()

    gaussσ = 1.


    FakeFibre, FakeWave = CreateToySlitlet(gaussσ) #Create the slitlets
    #FakeFibre, FakeWave = CreateToySlitlet(1.,0.,0.) #Create the slitlets

    noiseμ = maximum(FakeFibre)/10.
    noiseσ = noiseμ/10.
    outlier_frec = 0.01
    outlier_amp= 100.
    order = 2
    axemult = 1


    
    rm("ToyModelImages",force=true,recursive = true)
    mkdir("ToyModelImages")
    f = FITS("ToyModelImages/ContrastFibre.fits", "w")
    write(f,Contrast(FakeFibre,1))
    close(f)
    f = FITS("ToyModelImages/ContrastWave.fits", "w")
    write(f,Contrast(FakeWave,2))
    close(f)
    f = FITS("ToyModelImages/FakeFibre.fits", "w")
    write(f,FakeFibre)
    close(f)
    f = FITS("ToyModelImages/FakeWave.fits", "w")
    write(f,FakeWave)
    close(f)


    #Generate 2 2DPolynomials with defined coefiscients

    if order == 2
        refpix = (0,0)
        coef1 = [0.,0.,0.,1.005,0.05,0.]
        coef2 = [0.,0.,0.0001,0.008,1.,0.]
    end

    if order == 1
        refpix = (0,0)
        coef1 = [1.,0.05,0.]
        coef2 = [0.008,1.,0.]
    end

    #Adding noise
    AddNoise!(FakeFibre,noiseμ,noiseσ)
    AddNoise!(FakeWave,noiseμ,noiseσ)

    """
    #Adding outliers
    AddOutlier!(DeformedFakeFibre,outlier_frec,outlier_amp)
    AddOutlier!(DeformedFakeWave,outlier_frec,outlier_amp)
    """

    wgtmap = ones(size(FakeFibre)[1],size(FakeFibre)[2])

    Poly1 = Poly2D(order,refpix,coef1)
    Poly2 = Poly2D(order,refpix,coef2)

    #Adding margin
    (λmax,xmax) = size(FakeFibre)

    marginλ = convert(Int64,round(Poly1(λmax,xmax)- λmax+1.)) 
    marginx = convert(Int64,round(Poly2(λmax,xmax)- xmax+1.))

    FakeFibre = AddMargin(FakeFibre,marginλ,marginx)
    FakeWave= AddMargin(FakeWave,marginλ,marginx)



    #generate axes for warp transformation
    axess = MultAxes(axemult,FakeFibre)
    
    #Deform the images with ImageTransformations.warp 
    DeformedFakeFibre = ImageWarp(FakeFibre,Poly1,Poly2,axess)
    DeformedFakeWave = ImageWarp(FakeWave,Poly1,Poly2,axess)

    f = FITS("ToyModelImages/DeformedFakeFibreNoNoise.fits", "w")
    write(f,DeformedFakeFibre)
    close(f)
    f = FITS("ToyModelImages/DeformedFakeWaveNoNoise.fits", "w")
    write(f,DeformedFakeWave)
    close(f)

    


    """
    #Filtering hot pixels and noise 
    DeformedFakeFibre = ImgFilter(DeformedFakeFibreUnfiltered,3.,3.)
    DeformedFakeWave = ImgFilter(DeformedFakeWaveUnfiltered,3.,3.)
    """




    log = open("ToyModelImages/log.txt", "a")
    logtext = "Polynomials :

    refpix = $refpix
    order = $order
    gaussσ = $gaussσ
    noiseμ = $noiseμ
    noiseσ = $noiseσ
    outlier_frec = $outlier_frec
    outlier_amp = $outlier_amp
    marginλ = $marginλ
    marginx = $marginx
    axemult = $axemult
       
    coef1 = $coef1
    map1 = $(Poly1.map)

    coef2 = $coef2
    map2 = $(Poly2.map)"
    println(log,logtext)
    close(log)

    
    
    f = FITS("ToyModelImages/DeformedFakeFibre.fits", "w")
    write(f,DeformedFakeFibre)
    close(f)
    f = FITS("ToyModelImages/ResDeformedFakeFibre.fits", "w")
    write(f,Residual(FakeFibre,DeformedFakeFibre))
    close(f)
    f = FITS("ToyModelImages/DeformedFakeWave.fits", "w")
    write(f,DeformedFakeWave)
    close(f)
    f = FITS("ToyModelImages/ResDeformedFakeWave.fits", "w")
    write(f,Residual(FakeWave,DeformedFakeWave))
    close(f)

    """
    coef3 = [0.985429,0.0180358,0.0182765]
    coef4 = [-0.00981042,0.523253,0.00426099]
    Poly3 = Poly2D(order,refpix,coef3)
    Poly4 = Poly2D(order,refpix,coef4)
    ReformedFakeFibre = ImageWarp(DeformedFakeFibre,Poly3,Poly4,axess)
    f = FITS("ToyModelImages/ReformedFakeFibre.fits", "w")
    write(f,ReformedFakeFibre)
    close(f)
    """


    minimisation = MinCriteria(DeformedFakeFibre,DeformedFakeWave,wgtmap,order,refpix,FakeFibre,FakeWave;fitswriting = true)
    
    #Reinserting constant 
    #push!(Optim.minimizer(minimisation[1]),1.)
    #push!(Optim.minimizer(minimisation[1]),refpix[1])

    #insert!(Optim.minimizer(minimisation[2]),size(Optim.minimizer(minimisation[2]))[1],1.)
    #push!(Optim.minimizer(minimisation[2]),refpix[2])

    log = open("ToyModelImages/log.txt", "a")
    logtext = "
    
    Minimisation :

    $(minimisation)

    
    minimizer :
    
    $(Optim.minimizer(minimisation))
    
    minimimum :
    
    $(Optim.minimum(minimisation))
   
    wallclock duration : $(time() - timestart) seconds"

    println(log,logtext)
    close(log)


    reformcoef = vec(Optim.minimizer(minimisation))

    
    """
    log = open("ToyModelImages/log.txt", "a")
    logtext = "
    
    Minimisationfibre :

    (minimisation[1])

    Minimisationwave :

    (minimisation[2])
    
    minimizer :
    
    (Optim.minimizer(minimisation[2]))
    (Optim.minimizer(minimisation[1]))
    
    minimimum :
    
    (Optim.minimum(minimisation[2]))
    (Optim.minimum(minimisation[1]))
    
    wallclock duration : (time() - timestart) seconds"

    println(log,logtext)
    close(log)


    reformcoef = vec([Optim.minimizer(minimisation[2]) Optim.minimizer(minimisation[1])])
    """

    
    finaldeformedfibre, finaldeformedwave = ReformSlitlet(reformcoef,order,refpix,DeformedFakeFibre,DeformedFakeWave,FakeFibre,FakeWave)

    errorfibre = sum(Residual(FakeFibre,finaldeformedfibre).^2)/sum(Residual(FakeFibre,DeformedFakeFibre).^2)
    errorwave = sum(Residual(FakeWave,finaldeformedwave).^2)/sum(Residual(FakeWave,DeformedFakeWave).^2)
    log = open("ToyModelImages/log.txt", "a")
    logtext = "
    
    errorfibre = $errorfibre
    errorwave = $errorwave"

    println(log,logtext)
    println(logtext)
    close(log)


    return minimisation


end