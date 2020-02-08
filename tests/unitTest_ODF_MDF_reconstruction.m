classdef unitTest_ODF_MDF_reconstruction < matlab.unittest.TestCase
    
    properties ( TestParameter )
        r = {'/home/users/prasam46/mtex-5.1.1/'};
    end
  
    
    methods (Test)
       
        function testMtexpatherror(testCase)
            testCase.verifyError(@() ODF_reduction_algo('wrongPath',...
                200),'ODF_reduction_algo:e1')
        end
        
        function testFunctioninputs(testCase,r)
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200),'ODF_reduction_algo:e2')
        end 
        
        function testEBSDMatfileread(testCase,r)
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsdMatFile',...
            [r '/data' '/epidote.mat']),'ODF_reduction_algo:e3')
        end         
        
        function testInvalidOption1(testCase,r)
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'xxxx','xxxx'),...
            'ODF_reduction_algo:e5')
        end 
        
        function testEBSDInput(testCase,r)     
            
            f = load([r '/data' '/epidote.mat']);
            ebsd = f.ebsd;            
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd),...
            'ODF_reduction_algo:e3')
        end
        
        function testEBSDMultiInput(testCase,r)     
            
            f = load([r '/data' '/titanium.mat']);
            ebsd = f.ebsd;  
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd,'ebsdMatFile',...
            [r '/data' '/titanium.mat']),'ODF_reduction_algo:e4')
        end
            
        
        function testGrainsInput(testCase,r)     
            
            f = load([r '/data' '/epidote.mat']);
            ebsd = f.ebsd;
            grains = calcGrains(ebsd);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd('Glaucophane'),...
            'grains',grains),'ODF_reduction_algo:e6')
        end
        
        function testKernelInput(testCase,r)     
            f = load([r '/data' '/epidote.mat']);
            ebsd = f.ebsd;
            k = DirichletKernel(10);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd('Glaucophane'),...
            'kernel',k),'ODF_reduction_algo:e7')
        end
        
        function testGrainsMultiInput(testCase,r)     
            
            f = load([r '/data' '/titanium.mat']);
            ebsd = f.ebsd; 
            grains = calcGrains(ebsd);
            k = deLaValeePoussinKernel('halfwidth',5*degree);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd,'grains',...
            grains,'kernel',k),'ODF_reduction_algo:e4')
        end
               
        function testODFReconDemo(testCase,r)
            f = load([r '/data' '/alu.mat']);
            ebsd = f.ebsd;
            
            [temp1,temp2,temp3] = ODF_reduction_algo(r,200,'ebsd',ebsd);
            
            testCase.verifyClass(temp1 ,'orientation')

            testCase.verifyClass(temp2 ,'ODF')

            testCase.verifyClass(temp3 ,'double')

            testCase.verifyEqual(temp3,0.058,'AbsTol',0.001)
            
            testCase.verifyEqual(length(temp1),200)
            
        end
        
        function testMDFInput1(testCase,r)     
            
            f = load([r '/data' '/titanium.mat']);
            ebsd = f.ebsd;
            k = deLaValeePoussinKernel('halfwidth',5*degree);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd,'kernel',k,...
            'MisAngDist'),'ODF_reduction_algo:e8')
        end
        
        function testMDFInput2(testCase,r)     
            
            f = load([r '/data' '/alu.mat']);
            ebsd = f.ebsd;
            grains = calcGrains(ebsd);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd,'grains',grains,...
            'MisAngDist'),'ODF_reduction_algo:e10')
        end
        
        function testMDFInput3(testCase,r)     
            
            f = load([r '/data' '/alu.mat']);
            ebsd = f.ebsd;
            grains = calcGrains(ebsd);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,199,'ebsd',ebsd,'grains',grains,...
            'MisAngDist','sharedArea','shared_surfaceArea.csv',...
            'nbins',2.33),'ODF_reduction_algo:e11')
        end
        
        function testMDFVerifySharedArea(testCase,r)     
            
            f = load([r '/data' '/alu.mat']);
            ebsd = f.ebsd;
            grains = calcGrains(ebsd);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,200,'ebsd',ebsd,'grains',grains,...
            'MisAngDist','sharedArea','shared_surfaceArea.csv',...
            'nbins',12),'ODF_reduction_algo:e9')
        end
        
        
        function testMDFReconDemo(testCase,r)
            f = load([r '/data' '/alu.mat']);
            ebsd = f.ebsd;
            grains = calcGrains(ebsd);
            
            [temp1,~,~,temp2,temp3,temp4] = ODF_reduction_algo(r,199,'ebsd',...
                ebsd,'grains',grains,'MisAngDist','sharedArea',...
                'shared_surfaceArea.csv','nbins',12);
            
            testCase.verifyClass(temp1 ,'orientation')
            
            testCase.verifyClass(temp2 ,'double')

            testCase.verifyClass(temp3 ,'double')

            testCase.verifyClass(temp4 ,'double')
            
            testCase.verifyEqual(length(temp1) ,199)

            testCase.verifyEqual(length(temp2) ,12)

            testCase.verifyEqual(length(temp3) ,12)
            
            testCase.verifyEqual(length(temp4) ,12)
            
            testCase.verifyLessThan(0.5*sum(abs(temp2-temp3)),0.050)
            
        end
               
        
    end
    
  
    
    
end

