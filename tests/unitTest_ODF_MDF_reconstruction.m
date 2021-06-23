classdef unitTest_ODF_MDF_reconstruction < matlab.unittest.TestCase
    
    properties ( TestParameter )
        r = {'/home/users/biswaa5w/mtex-5.5.2/'};
    end
  
    
    methods (Test)
       
        function testMtexpatherror(testCase)
            testCase.verifyError(@() ODF_reduction_algo('wrongPath',...
                40),'ODF_reduction_algo:e1')
        end
        
        function testFunctioninputs(testCase,r)
            testCase.verifyError(...
            @() ODF_reduction_algo(r,40),'ODF_reduction_algo:e2')
        end 
          
        
        function testInvalidOption1(testCase,r)
            testCase.verifyError(...
            @() ODF_reduction_algo(r,40,'xxxx','xxxx'),...
            'ODF_reduction_algo:e5')
        end 
        
        
        function testEBSDMultiInput(testCase,r)     
            
            ebsd = importdata('ebsd_316L.mat');              
            testCase.verifyError(...
            @() ODF_reduction_algo(r,40,'ebsd',ebsd,'ebsdMatFile',...
            'ebsd_316L.mat'),'ODF_reduction_algo:e4')
        end
            
        
        
        function testKernelInput(testCase,r)     
            ebsd = importdata('ebsd_316L.mat');            
            k = DirichletKernel(10);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,40,'ebsd',ebsd,...
            'kernel',k),'ODF_reduction_algo:e7')
        end
        
               
        function testODFReconDemo(testCase,r)
            ebsd = importdata('ebsd_316L.mat');            
            
            [temp1,temp2,temp3] = ODF_reduction_algo(r,40,'ebsd',ebsd);
            
            testCase.verifyClass(temp1 ,'orientation')

            testCase.verifyClass(temp2 ,'ODF')

            testCase.verifyClass(temp3 ,'double')

            testCase.verifyEqual(temp3,0.23053,'AbsTol',0.001)
            
            testCase.verifyEqual(length(temp1),40)
            
        end
        
        function testMDFInput1(testCase,r)     
            
            ebsd = importdata('ebsd_316L.mat');  
            try
                k = deLaValeePoussinKernel('halfwidth',5*degree);
            catch
                warning('off')
                psi = deLaValleePoussinKernel('halfwidth',5*degree);
            end
            testCase.verifyError(...
            @() ODF_reduction_algo(r,40,'ebsd',ebsd,...
            'MisAngDist'),'ODF_reduction_algo:e8')
        end
        
        function testMDFInput2(testCase,r)     
            
            ebsd = importdata('ebsd_316L.mat');  
            grains = importdata('grains_316L.mat');
            testCase.verifyError(...
            @() ODF_reduction_algo(r,40,'ebsd',ebsd,'grains',grains,...
            'MisAngDist'),'ODF_reduction_algo:e10')
        end
        
        function testMDFInput3(testCase,r)     
            
            ebsd = importdata('ebsd_316L.mat');
            grains = calcGrains(ebsd);
            testCase.verifyError(...
            @() ODF_reduction_algo(r,40,'ebsd',ebsd,'grains',grains,...
            'MisAngDist','sharedArea','shared_surfaceArea.csv',...
            'nbins',2.33),'ODF_reduction_algo:e11')
        end
        
        function testMDFVerifySharedArea(testCase,r)     
            
            ebsd = importdata('ebsd_316L.mat');
            grains = importdata('grains_316L.mat');
            testCase.verifyError(...
            @() ODF_reduction_algo(r,100,'ebsd',ebsd,'grains',grains,...
            'MisAngDist','sharedArea','shared_surfaceArea.csv',...
            'nbins',12),'ODF_reduction_algo:e9')
        end
        
        
        function testMDFReconDemo(testCase,r)
            ebsd = importdata('ebsd_316L.mat');
            grains = importdata('grains_316L.mat');
            
            [temp1,~,~,temp2,temp3,temp4] = ODF_reduction_algo(r,40,'ebsd',...
                ebsd,'grains',grains,'MisAngDist','sharedArea',...
                'shared_surfaceArea.csv','nbins',12);
            
            testCase.verifyClass(temp1 ,'orientation')
            
            testCase.verifyClass(temp2 ,'double')

            testCase.verifyClass(temp3 ,'double')

            testCase.verifyClass(temp4 ,'double')
            
            testCase.verifyEqual(length(temp1) ,40)

            testCase.verifyEqual(length(temp2) ,12)

            testCase.verifyEqual(length(temp3) ,12)
            
            testCase.verifyEqual(length(temp4) ,12)
            
            testCase.verifyLessThan(0.5*sum(abs(temp2-temp3)),0.10)
            
        end
               
        
    end
    
  
    
    
end

