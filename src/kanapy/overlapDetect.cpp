#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>
#include <vector>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXf;
using Eigen::ArrayXXf;

bool collideDetect (MatrixXf coef_i, MatrixXf coef_j, MatrixXf r_i, MatrixXf r_j, MatrixXf A_i, MatrixXf A_j) {          
    
    // Initialize Matrices A & B with zeros
    MatrixXf A = MatrixXf::Zero(4, 4);   
    MatrixXf B = MatrixXf::Zero(4, 4);
    
    A(0,0) = 1/pow(coef_i(0), 2);
    A(1,1) = 1/pow(coef_i(1), 2);
    A(2,2) = 1/pow(coef_i(2), 2);
    A(3,3) = -1;
    
    B(0,0) = 1/pow(coef_j(0), 2);
    B(1,1) = 1/pow(coef_j(1), 2);
    B(2,2) = 1/pow(coef_j(2), 2);
    B(3,3) = -1;
            
        
    // Rigid body transformations
    MatrixXf T_i = MatrixXf::Zero(4, 4); 
    MatrixXf T_j = MatrixXf::Zero(4, 4);  
    
    T_i.block<3,3>(0,0) = A_i;
    T_i.block<3,1>(0,3) = r_i;
    T_i.row(3) << 0.0,0.0,0.0,1.0;
   
    T_j.block<3,3>(0,0) = A_j;
    T_j.block<3,1>(0,3) = r_j;
    T_j.row(3) << 0.0,0.0,0.0,1.0;
    
    // Copy the arrays for future operations
    MatrixXf Ma = MatrixXf::Zero(4, 4); 
    MatrixXf Mb = MatrixXf::Zero(4, 4); 

    Ma = T_i.replicate(1, 1);
    Mb = T_j.replicate(1, 1);    
    
    // aij of matrix A in det(lambda*A - Ma'*(Mb^-1)'*B*(Mb^-1)*Ma).        
    // bij of matrix b = Ma'*(Mb^-1)'*B*(Mb^-1)*Ma
    MatrixXf aux = MatrixXf::Zero(4, 4); 
    MatrixXf b = MatrixXf::Zero(4, 4); 

    aux = Mb.inverse() * Ma;   
    b = aux.transpose() * B * aux;

    
    // Coefficients of the Characteristic Polynomial.
    double T0, T1, T2, T3, T4;
    T4 = (-A(0,0)*A(1,1)*A(2,2));
    T3 = (A(0,0)*A(1,1)*b(2,2) + A(0,0)*A(2,2)*b(1,1) + A(1,1)*A(2,2)*b(0,0) - A(0,0)*A(1,1)*A(2,2)*b(3,3));
    T2 = (A(0,0)*b(1,2)*b(2,1) - A(0,0)*b(1,1)*b(2,2) - A(1,1)*b(0,0)*b(2,2) + A(1,1)*b(0,2)*b(2,0) - 
          A(2,2)*b(0,0)*b(1,1) + A(2,2)*b(0,1)*b(1,0) + A(0,0)*A(1,1)*b(2,2)*b(3,3) - A(0,0)*A(1,1)*b(2,3)*b(3,2) + 
          A(0,0)*A(2,2)*b(1,1)*b(3,3) - A(0,0)*A(2,2)*b(1,3)*b(3,1) + A(1,1)*A(2,2)*b(0,0)*b(3,3) - 
          A(1,1)*A(2,2)*b(0,3)*b(3,0));
    T1 = (b(0,0)*b(1,1)*b(2,2) - b(0,0)*b(1,2)*b(2,1) - b(0,1)*b(1,0)*b(2,2) + b(0,1)*b(1,2)*b(2,0) + 
          b(0,2)*b(1,0)*b(2,1) - b(0,2)*b(1,1)*b(2,0) - A(0,0)*b(1,1)*b(2,2)*b(3,3) + A(0,0)*b(1,1)*b(2,3)*b(3,2) + 
          A(0,0)*b(1,2)*b(2,1)*b(3,3) - A(0,0)*b(1,2)*b(2,3)*b(3,1) - A(0,0)*b(1,3)*b(2,1)*b(3,2) + 
          A(0,0)*b(1,3)*b(2,2)*b(3,1) - A(1,1)*b(0,0)*b(2,2)*b(3,3) + A(1,1)*b(0,0)*b(2,3)*b(3,2) + 
          A(1,1)*b(0,2)*b(2,0)*b(3,3) - A(1,1)*b(0,2)*b(2,3)*b(3,0) - A(1,1)*b(0,3)*b(2,0)*b(3,2) + 
          A(1,1)*b(0,3)*b(2,2)*b(3,0) - A(2,2)*b(0,0)*b(1,1)*b(3,3) + A(2,2)*b(0,0)*b(1,3)*b(3,1) + 
          A(2,2)*b(0,1)*b(1,0)*b(3,3) - A(2,2)*b(0,1)*b(1,3)*b(3,0) - A(2,2)*b(0,3)*b(1,0)*b(3,1) + 
          A(2,2)*b(0,3)*b(1,1)*b(3,0));
    T0 = (b(0,0)*b(1,1)*b(2,2)*b(3,3) - b(0,0)*b(1,1)*b(2,3)*b(3,2) - b(0,0)*b(1,2)*b(2,1)*b(3,3) + 
          b(0,0)*b(1,2)*b(2,3)*b(3,1) + b(0,0)*b(1,3)*b(2,1)*b(3,2) - b(0,0)*b(1,3)*b(2,2)*b(3,1) - 
          b(0,1)*b(1,0)*b(2,2)*b(3,3) + b(0,1)*b(1,0)*b(2,3)*b(3,2) + b(0,1)*b(1,2)*b(2,0)*b(3,3) - 
          b(0,1)*b(1,2)*b(2,3)*b(3,0) - b(0,1)*b(1,3)*b(2,0)*b(3,2) + b(0,1)*b(1,3)*b(2,2)*b(3,0) + 
          b(0,2)*b(1,0)*b(2,1)*b(3,3) - b(0,2)*b(1,0)*b(2,3)*b(3,1) - b(0,2)*b(1,1)*b(2,0)*b(3,3) + 
          b(0,2)*b(1,1)*b(2,3)*b(3,0) + b(0,2)*b(1,3)*b(2,0)*b(3,1) - b(0,2)*b(1,3)*b(2,1)*b(3,0) - 
          b(0,3)*b(1,0)*b(2,1)*b(3,2) + b(0,3)*b(1,0)*b(2,2)*b(3,1) + b(0,3)*b(1,1)*b(2,0)*b(3,2) - 
          b(0,3)*b(1,1)*b(2,2)*b(3,0) - b(0,3)*b(1,2)*b(2,0)*b(3,1) + b(0,3)*b(1,2)*b(2,1)*b(3,0));
    
    //  Roots of the characteristic_polynomial (lambda0, ... , lambda4).
    Eigen::Matrix<double,5,1> cp;
    cp << T0, T1, T2, T3, T4;
    
    // Solve the polynomial       
    PolynomialSolver<double,4> psolve( cp );
    
    // Find the real roots where imaginary part does'nt exist
    double realRoots[4];
    for (int i = 0; i < 4; i++) {         
        std::complex<double> mycomplex = psolve.roots()[i];
        if (mycomplex.imag() == 0.0)
            realRoots[i] = mycomplex.real();
    }
          
    // Count number of real negative roots
    int count_neg = 0;
    for (int i = 0; i < 4; i++) {         
        if (realRoots[i] < 0)
            count_neg += 1;
    }
    
    // Sort the real roots in ascending order
    int n = sizeof(realRoots)/sizeof(realRoots[0]);   
    sort(realRoots, realRoots+n);                       // sorts in ascending order
    //sort(realRoots, realRoots+n, greater<int>());       // sorts in descending order    

    // Algebraic separation conditions to determine overlapping
    if (count_neg == 2){
        if (realRoots[0] != realRoots[1]){
            //status = 'separated'
            return false;
        }
        else if (abs(realRoots[0] - realRoots[1]) <= 0.01){
            //status = 'touching'
            return true;
        }
        else{
            return true;        
        }

    }
    else{
        //status = 'overlapping'
        return true;
    }    


}

namespace py = pybind11;

PYBIND11_MODULE(base, m) {

    m.doc() = R"pbdoc(
        Pybind11 plugin
        ---------------
        The lightweight header-only library pybind11 is used to create Python bindings for the code written in C++.        
        The function can be complied individually using the command documented here_.
        
        .. _here: https://pybind11.readthedocs.io/en/stable/compiling.html
        
    )pbdoc";
    
    m.def("collideDetect", &collideDetect, R"pbdoc(
        C++ implementation of Algebraic separation condition developed by W. Wang et al. 2001 for overlap detection
        between two static ellipsoids.
        
        :param arg0: Coefficients of ellipsoid :math:`i`
        :type arg0: numpy array
        :param arg1: Coefficients of ellipsoid :math:`j`
        :type arg1: numpy array
        :param arg2: Position of ellipsoid :math:`i`
        :type arg2: numpy array
        :param arg3: Position of ellipsoid :math:`j`
        :type arg3: numpy array
        :param arg4: Rotation matrix of ellipsoid :math:`i`
        :type arg4: numpy array
        :param arg5: Rotation matrix of ellipsoid :math:`j`
        :type arg5: numpy array
        :returns: **True** if ellipoids :math:`i, j` overlap, else **False**
        :rtype: boolean             
        
        .. note::    Ellipsoids in their standard form in their local coordinate systems are given by,    

                     .. image:: /figs/ell_std_form.png                        
                        :width: 400px
                        :height: 75px
                        :align: center
                
                     where :math:`a^i, a^j` are the semi-major axis lengths, :math:`b^i, b^j` and :math:`c^i, c^j` are the semi-minor axes lengths. :math:`\mathbf{A}^{*i}` 
                     and :math:`\mathbf{A}^{'j}` in matrix form is given by,            

                     .. image:: /figs/ell_matrix_form.png
                        :width: 550px
                        :height: 120px
                        :align: center
                    
                     The transformation from local coordinate systems  :math:`\mathbf{e}_1^{*}\mathbf{e}_2^{*}\mathbf{e}_3^{*}` and :math:`\mathbf{e}_1^{'}\mathbf{e}_2^{'}\mathbf{e}_3^{'}` 
                     to global coordinate system :math:`\mathbf{E}_1\mathbf{E}_2\mathbf{E}_3` is represented by,
                     :math:`\mathbf{X} = \mathbf{M}^{i} \:\mathbf{X}^{*}` and :math:`\mathbf{X} = \mathbf{M}^{j} \:\mathbf{X}^{'}`
                    
                     where :math:`\mathbf{M}^{i}` and :math:`\mathbf{M}^{j}` are transformation matrices that contain both rotation and translation components.                        

                     .. image:: /figs/transformation_matrix.png
                        :width: 300px
                        :height: 45px
                        :align: center
                        
                     The equation of ellipsoids in global coordinate system is now given as: :math:`\mathbf{X}^{T} \mathbf{A}^{i} \mathbf{X}` and :math:`\mathbf{X}^{T} 
                     \mathbf{A}^{j} \mathbf{X}` where,

                     .. image:: /figs/ell_global_eqn.png           
                        :width: 230px
                        :height: 50px
                        :align: center
                        
                     The characteristic equation can now be written in the form,           

                     .. image:: /figs/characteristic_eqn.png          
                        :width: 300px
                        :height: 50px
                        :align: center
                        
                     The fourth order polynomial is solved and depending on the nature of the roots obtained the overlap or separation conditions between 
                     ellipsoids can be established as described in :ref:`Overlap detection`.
                                        
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";        
#endif
}
