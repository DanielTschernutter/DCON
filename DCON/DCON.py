import numpy as np
from sklearn.linear_model import Ridge
import osqp
import random
from scipy import sparse
from tensorflow import keras

class DCON():
    def __init__(self,n_hidden,n_inputs,conv_tol=1e-6,DC_stopping_tol=1e-4):
        
        if not isinstance(n_hidden, int):
            raise ValueError('Number of hidden neurons n_hidden should be of type int.')
        else:
            if n_hidden<1:
                raise ValueError('Number of hidden neurons n_hidden should be at least one.')
            else:
                self.n_hidden = n_hidden
        
        if not isinstance(n_inputs, int):
            raise ValueError('Number of inputs n_inputs should be of type int.')
        else:
            if n_inputs<1:
                raise ValueError('Number of inputs n_inputs should be at least one.')
            else:
                self.n_inputs = n_inputs

        if not isinstance(conv_tol, float):
            raise ValueError('Parameter for convergence tolerance conv_tol should be of type float.')
        else:
            if conv_tol<=0.0:
                raise ValueError('Parameter for convergence tolerance conv_tol should be greater than zero.')
            else:
                self.conv_tol = conv_tol

        if not isinstance(DC_stopping_tol, float):
            raise ValueError('Parameter for DCA convergence DC_stopping_tol should be of type float.')
        else:
            if DC_stopping_tol<=0.0:
                raise ValueError('Parameter for DCA convergence DC_stopping_tol should be greater than zero.')
            else:
                self.DC_stopping_tol = DC_stopping_tol

        self.model_fitted = False
        self.keras_model = None        
        self.mse = None
        self.norm_diffs = None
        
    def _check_data(self, arr, n_dims, first_dim=False, second_dim=False, name=''):
        # Check if numpy array
        if not isinstance(arr, np.ndarray):
            raise ValueError('Input {} is not a numpy array'.format(name))
            
        # Check datatype
        if not arr.dtype == 'float64':
            raise ValueError('Input array {} not of dtype float64'.format(name))

        # Check nan
        if np.isnan(arr).any():
            raise ValueError('Input array {} containing nan'.format(name))
        
        # Check infinity
        if np.isinf(arr).any():
            raise ValueError('Input array {} containing infinity'.format(name))
        
        # Check dimensions
        if len(arr.shape) != n_dims:
            raise ValueError('Input array {} having {} dimensions,'
                             ' while {} are required'.format(name, len(arr.shape), n_dims))
        
        if first_dim != False:
            if arr.shape[0] != first_dim:
                raise ValueError('First dimension of {} should be {}'.format(name, first_dim))
        
        if second_dim != False:
            if arr.shape[1] != second_dim:
                raise ValueError('Second dimension of {} should be {}'.format(name, second_dim))
    
    def _update_betas(self,subproblem_index):
        # Index set I_1
        for j in range(self.num_samples):
            if self.y[j]*self.alpha[subproblem_index]<0:
                self.beta_g[j]=abs(2.0*self.y[j]*self.alpha[subproblem_index])
                self.beta_h[j]=0.0
            else:
                self.beta_g[j]=0.0
                self.beta_h[j]=2.0*self.y[j]*self.alpha[subproblem_index]
        
        # Index set I_2
        for j in range(self.num_samples):
            for i in range(subproblem_index+1,self.n_hidden):
                val=self.alpha[i]*self.alpha[subproblem_index]*max(np.dot(self.W[:,i],self.X[j,:])+self.B[i],0.0)
                if val>=0:
                    self.beta_g[j]+=2.0*val
                else:
                    self.beta_h[j]+=2.0*abs(val)
        
        # Index set I_3
        for j in range(self.num_samples):
            for k in range(subproblem_index):
                val=self.alpha[subproblem_index]*self.alpha[k]*max(np.dot(self.W[:,k],self.X[j,:])+self.B[k],0.0)
                if val>=0:
                    self.beta_g[j]+=2.0*val
                else:
                    self.beta_h[j]+=2.0*abs(val)
        
    def _update_Sigma(self, validation=False):
        if not validation:
            self.Sigma=self.X @ self.W + np.repeat(self.B.reshape(1,-1),self.num_samples,axis=0)
            self.Sigma=self.Sigma*(self.Sigma>0)
        else:
            self.Sigma=self.X_val @ self.W + np.repeat(self.B.reshape(1,-1),self.num_samples_val,axis=0)
            self.Sigma=self.Sigma*(self.Sigma>0)
        
    def _get_mse(self, validation=False):
        # Sigma has to be updated with _update_Sigma to get expected results from this function
        if not validation:
            return (np.linalg.norm(self.y - self.Sigma @ self.alpha)**2)/self.num_samples
        else:
            return (np.linalg.norm(self.y_val - self.Sigma @ self.alpha)**2)/self.num_samples_val
        
    def _update_QP(self,subproblem_index,w,b):
        # Build matrices only in first call then only update
        if self.QP is None:
            self.QP=osqp.OSQP()
            
            # Init P
            zero_n_plus_1_m=sparse.csc_matrix((self.n_inputs+1,self.num_samples))
            zero_m_n_plus_1=sparse.csc_matrix((self.num_samples,self.n_inputs+1))
            zero_n_plus_1_n_plus_1=sparse.csc_matrix((self.n_inputs+1,self.n_inputs+1))
            P=sparse.bmat([[2.0*(1+self.reg_param)*sparse.eye(self.num_samples), 2.0*(-self.reg_param)*sparse.eye(self.num_samples) , zero_m_n_plus_1 , zero_m_n_plus_1],
                           [2.0*(-self.reg_param)*sparse.eye(self.num_samples), 2.0*self.reg_param*sparse.eye(self.num_samples), zero_m_n_plus_1 , zero_m_n_plus_1],
                           [zero_n_plus_1_m , zero_n_plus_1_m , zero_n_plus_1_n_plus_1 , zero_n_plus_1_n_plus_1],
                           [zero_n_plus_1_m , zero_n_plus_1_m , zero_n_plus_1_n_plus_1 , zero_n_plus_1_n_plus_1]])
            P=P.tocsc()
            
            # Init q
            q=np.zeros(2*self.num_samples+2*(self.n_inputs+1))
            
            # Init A
            I_m_m=sparse.eye(self.num_samples)
            I_n_plus_1_n_plus_1=sparse.eye(self.n_inputs+1)
            M=np.hstack([self.X, np.ones([self.X.shape[0], 1])])
            A=sparse.bmat([[-I_m_m,I_m_m,M,-M],
                           [I_m_m,-I_m_m,-M,M],
                           [I_m_m,None,None,None],
                           [None,I_m_m,None,None],
                           [None,None,I_n_plus_1_n_plus_1,None],
                           [None,None,None,I_n_plus_1_n_plus_1]])
            A=A.tocsc()
            
            # Init l and u    
            l=np.zeros(4*self.num_samples+2*(self.n_inputs+1))
            u=np.full((4*self.num_samples+2*(self.n_inputs+1),),np.inf)
            
            # Init QP
            if self.verbose==2:
                self.QP.setup(P=P, q=q, A=A, l=l, u=u, verbose=True, eps_abs=self.DC_stopping_tol/10, eps_rel=self.DC_stopping_tol/10)    
            else:
                self.QP.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=self.DC_stopping_tol/10, eps_rel=self.DC_stopping_tol/10)
        
        ## Update elements of QP
        
        self._update_betas(subproblem_index)
        # Update q
        q=np.zeros(2*self.num_samples+2*(self.n_inputs+1))
        for j in range(self.num_samples):
            val=np.dot(self.X[j,:],w)+b
            if val>=0:
                q[j]=self.beta_g[j]-self.beta_h[j]
                q[self.num_samples+j]=self.beta_h[j]
            else:
                q[j]=self.beta_g[j]
        self.QP.update(q=q)
        
        # Update P
        new_values=np.array(self.num_samples*[2.0*(self.alpha[subproblem_index]**2+self.reg_param)])
        new_indices=np.array(range(0,2*self.num_samples,2))
        self.QP.update(Px=new_values, Px_idx=new_indices)
        
    def _eval_g(self,w,b,alpha_subproblem):
        # Make sure _update_betas is called before using this function
        return sum([self.beta_g[j]*max(np.dot(w,self.X[j,:])+b,0.0)+
                    alpha_subproblem**2*max(np.dot(w,self.X[j,:])+b,0.0)**2+
                    self.reg_param*(np.dot(w,self.X[j,:])+b)**2 for j in range(self.num_samples)])/self.num_samples
        
    def _eval_h(self,w,b):
        # Make sure _update_betas is called before using this function
        return sum([self.beta_h[j]*max(np.dot(w,self.X[j,:])+b,0.0) for j in range(self.num_samples)])/self.num_samples
    
    def _solve_DC_subproblem(self,subproblem_index,subproblem_number):
        self._update_betas(subproblem_index)
        w=self.W[:,subproblem_index]
        b=self.B[subproblem_index]
        
        # Note that osqp performs automatic warm starts with previous solutions if we only update
        norm_diff=2*self.DC_stopping_tol
        DC_iter=0
        loss_before=self._eval_g(w,b,self.alpha[subproblem_index])-self._eval_h(w,b)
        while norm_diff>=self.DC_stopping_tol and DC_iter<self.max_num_DC_iterations:
            self._update_QP(subproblem_index,w,b)
            if self.verbose==2:
                print("")
                print("Solving QP...")
                print("")
            results = self.QP.solve()
            v3=results.x[-2*(self.n_inputs+1):-(self.n_inputs+1)]
            v4=results.x[-(self.n_inputs+1):]
            w_new=np.array(v3[:-1]-v4[:-1])
            b_new=v3[-1]-v4[-1]
            norm_diff=np.linalg.norm(np.hstack([w_new-w,np.array(b_new-b)]))
            w=w_new
            b=b_new
            DC_iter+=1
            if self.verbose>=1:
                if self.current_norm_diff_outer is not None:
                    output_string="Iteration {} (DC Subproblem {} of {}, DC Iteration {}): Outer Norm Diff={:.4f}, Inner Norm Diff={:.4f}".format(self.iteration_count+1,subproblem_number,self.n_hidden,DC_iter,self.current_norm_diff_outer,norm_diff)+20*" "
                    print("\r"+output_string,end='')
                    self.max_string_length=max(len(output_string),self.max_string_length)
                else:
                    output_string="Iteration {} (DC Subproblem {} of {}, DC Iteration {}): Outer Norm Diff={}, Inner Norm Diff={:.4f}".format(self.iteration_count+1,subproblem_number,self.n_hidden,DC_iter,self.current_norm_diff_outer,norm_diff)+20*" "
                    print("\r"+output_string,end='')
                    self.max_string_length=len(output_string)
                if self.verbose==2:
                    print("")
              
        loss_after=self._eval_g(w,b,self.alpha[subproblem_index])-self._eval_h(w,b)
        if loss_before<loss_after:
            w=self.W[:,subproblem_index]
            b=self.B[subproblem_index]
            if self.verbose==2:
                print("DC step ignored due to inexact numerics in QP solver.")
        
        norm_squared_w_b=np.linalg.norm(self.W[:,subproblem_index]-w)**2
        self.W[:,subproblem_index]=w
        norm_squared_w_b+=(self.B[subproblem_index]-b)**2
        self.B[subproblem_index]=b
        
        return norm_squared_w_b
        
    def _solve_Alpha_subproblem(self):
        old_alpha=self.alpha
        self._update_Sigma()
        
        loss_before=self._get_mse()+(self.reg_param/self.num_samples)*np.linalg.norm(self.alpha)**2
        
        if self.Ridge is None:
            self.Ridge=Ridge(alpha=self.reg_param,fit_intercept=False,random_state=self.random_seed)
        
        self.Ridge=self.Ridge.fit(self.Sigma,self.y)
        norm_squared_alpha=np.linalg.norm(self.alpha - self.Ridge.coef_)**2
        self.alpha = self.Ridge.coef_
        loss_after=self._get_mse()+(self.reg_param/self.num_samples)*np.linalg.norm(self.alpha)**2
        
        if loss_before<loss_after:
            self.alpha=old_alpha
            norm_squared_alpha=0
            if self.verbose==2:
                print("Alpha step ignored due to inexact numerics in subproblem solver.")
        
        return norm_squared_alpha
        
    def fit(self, X, y, n_epochs, W=False, B=False, alpha=False, reg_param=0.1, max_num_DC_iterations=100, verbose=1, random_seed=0, validation_data=None, patience_early_stopping=10):
        """
        ===
        fit
        ===
        
        Definition: fit(self, X, y, n_epochs, W=False, B=False, alpha=False, reg_param=0.1, max_num_DC_iterations=100, verbose=1, random_seed=0, validation_data=None, patience_early_stopping=10)
            
        Type: Method of DCON class
        
        This method fits the SLFN to the dataset provided.
        
        Parameters
        ----------
        X : np.array
            Feature matrix
            
        y : np.array
            Corresponding labels
            
        n_epochs : int
            Maximal number of epochs to run, if set to 'auto' DCON will run until convergence (distance between iterates is smaller than self.conv_tol) or if validation data is provided until termination by early stopping
            
        W : np.array, default = False
            If provided, matrix W (shape == (self.n, self.N)) of the neural network will be initialized accordingly
            
        B : np.array, default = False
            If provided, matrix B (shape == (self.N,)) of the neural network will be initialized accordingly
            
        alpha : np.array, default = False
            If provided, matrix alpha (shape == (self.N,)) of the neural network will be initialized accordingly
            
        reg_param : float, default = 0.1
            Regularization parameter for training
            
        max_num_DC_iterations : int, default = 100
            Maximum number of DC iterations to run
            
        verbose : int, default = 1
            Parameter to control the output of DCON
            
        random_seed : int, default = 0
            Random seed
            
        validation_data : (np.array,np.array), default = None
            If validation data is provided in the form (X_val,y_val) the .fit routine performs an early stopping procedure.
            
        patience_early_stopping : int, default = 10
            If validation data is provided in the form (X_val,y_val) the .fit routine performs an early stopping procedure. This parameter controlls the maximum number of epochs without progress on the validation data.
        """

        ## Checks and inits
        
        # Set random seed
        np.random.seed(random_seed)
        np.random.RandomState(random_seed)

        # Check X        
        self._check_data(X, n_dims=2, second_dim=self.n_inputs, name='X')
        
        # Check y
        if isinstance(y, np.ndarray):
            y = np.squeeze(y)
        else:    
            raise ValueError('Input y is not a numpy array')
        self._check_data(y, n_dims=1, name='y')
        
        # Check lengths
        if len(X) != len(y):
            raise ValueError('X and y do not have the same number of samples')
        else:
            self.num_samples=len(y)
            
        # Check or init W,B, and alpha
        if isinstance(W, np.ndarray):
            self._check_data(W, n_dims=2, first_dim=self.n_inputs, second_dim=self.n_hidden, name='W')
        else:
            # Initialize weights W
            W = np.random.uniform(low=-1.0, high=1.0, size=(self.n_inputs, self.n_hidden))*(np.sqrt(6)/np.sqrt(self.n_inputs+self.n_hidden)) # Xavier initialization
        if isinstance(B, np.ndarray):
            self._check_data(B, n_dims=1, first_dim=self.n_hidden, name='B')
        else:
            # Initialize weights B
            B = np.zeros(self.n_hidden) # Xavier initialization
        if isinstance(alpha, np.ndarray):
            self._check_data(alpha, n_dims=1, first_dim=self.n_hidden, name='alpha')
        else:
            # Initialize weights alpha
            alpha = np.random.uniform(low=-1.0, high=1.0, size=(self.n_hidden))*(np.sqrt(6)/np.sqrt(self.n_hidden+1)) # Xavier initialization
        
        # Check validation data
        if validation_data is not None:
            if isinstance(validation_data,tuple) and len(validation_data)==2:
                self._check_data(validation_data[0], n_dims=2, second_dim=self.n_inputs, name='X_val')
                self.X_val = validation_data[0]
                
                if isinstance(validation_data[1], np.ndarray):
                    y_val = validation_data[1]
                else:    
                    raise ValueError('Input y_val is not a numpy array')
                self._check_data(y_val, n_dims=1, name='y_val')
                self.y_val = y_val
                
                if len(self.X_val) != len(self.y_val):
                    raise ValueError('X_val and y_val do not have the same number of samples')
                else:
                    self.num_samples_val=len(self.y_val)
            else:
                raise ValueError('validation_data must be a tuple of length 2')
        else:
            self.X_val = None
            self.y_val = None
        
        # Additional checks
        if not isinstance(reg_param, float):
            raise ValueError('Parameter reg_param should be of type float')
        if not isinstance(max_num_DC_iterations, int):
            raise ValueError('Parameter max_num_DC_iterations should be of type int')
        if max_num_DC_iterations<1:
            raise ValueError('Parameter max_num_of_DC_iterations should be at least 1')
        if not isinstance(verbose, int):
            raise ValueError('Parameter verbose should be int')
        if verbose not in [0,1,2]:
            raise ValueError('Parameter verbose should be in [0,1,2]')
        if reg_param<=0.0:
            raise ValueError('Parameter reg_param should be greater than zero')
        if n_epochs=='auto':
            n_epochs=np.inf
        else:
            if not isinstance(n_epochs, int):
                raise ValueError('Parameter n_epochs should be of type int or set to auto')
            if n_epochs<1:
                raise ValueError('Parameter n_epochs should be at least 1')
        if not isinstance(patience_early_stopping, int):
            raise ValueError('Parameter patience_early_stopping should be of type int')
        if patience_early_stopping<=0:
            raise ValueError('Parameter patience_early_stopping should be greater than zero')
                
        # Set variables
        self.X=X
        self.y=y
        self.n_epochs=n_epochs
        self.W=W
        self.B=B
        self.alpha=alpha
        self.reg_param=reg_param
        self.max_num_DC_iterations=max_num_DC_iterations
        self.verbose=verbose
        self.random_seed=random_seed
        self.patience_early_stopping=patience_early_stopping
        
        # Init additional variables
        self.beta_g=np.zeros(self.num_samples)
        self.beta_h=np.zeros(self.num_samples)
        self.Sigma=np.zeros((self.num_samples,self.n_hidden))
        self.QP=None
        self.Ridge=None
        self.current_norm_diff_outer=None
        self.iteration_count=0
        self.mse=[]
        self.mse_val=[]
        self.norm_diffs=[]
        self.max_string_length=None
        
        ## DCON Algorithm
        
        permutation=list(range(self.n_hidden))
        early_stopping_count=0
        min_val_loss=np.inf
        optimal_params=None
        
        while self.iteration_count<self.n_epochs:
            
            random.Random(self.random_seed+self.iteration_count).shuffle(permutation)
            norm_squared=0
            for subproblem_number,subproblem_index in enumerate(permutation):
                if abs(self.alpha[subproblem_index])>=1e-10:
                    norm_squared+=self._solve_DC_subproblem(subproblem_index,subproblem_number)
                else:
                    norm_squared+=np.linalg.norm(self.W[:,subproblem_index])**2
                    self.W[:,subproblem_index]=0
                    norm_squared+=(self.B[subproblem_index])**2
                    self.B[subproblem_index]=0
                    if self.verbose==2:
                        print("Iteration "+str(self.iteration_count+1)+" (DC Subproblem "+str(subproblem_number)+" of "+str(self.n_hidden)+"): skipped as alpha["+str(subproblem_index)+"]=0. Setting w and b to zero.")
            norm_squared+=self._solve_Alpha_subproblem()
            self.current_norm_diff_outer=np.sqrt(norm_squared)
            
            if self.verbose>=1:
                print("\rIteration {} (Alpha Subproblem): Outer Norm Diff={:.4f}".format(self.iteration_count+1,self.current_norm_diff_outer)+self.max_string_length*" ",end='')
                print(" ")
                
            self.iteration_count+=1
            
            self.mse.append(self._get_mse())
            self.norm_diffs.append(self.current_norm_diff_outer)
            
            if self.X_val is None:
                if self.current_norm_diff_outer<self.conv_tol:
                    if self.verbose>=1:
                        print("Convergence after {} iteration(s).".format(self.iteration_count))
                    break
            else:
                self._update_Sigma(validation=True)
                val_loss = self._get_mse(validation=True)
                self.mse_val.append(val_loss)
                if val_loss<min_val_loss:
                    min_val_loss = val_loss
                    early_stopping_count = 0
                    optimal_params={'W': self.W, 'B': self.B, 'alpha': self.alpha}
                    if self.verbose>=1:
                        print("New best validation loss ({:.4f}) after {} iteration(s).".format(min_val_loss,self.iteration_count))
                else:
                    early_stopping_count+=1
                if early_stopping_count==self.patience_early_stopping:
                    self.W=optimal_params['W']
                    self.B=optimal_params['B']
                    self.alpha=optimal_params['alpha']
                    if self.verbose>=1:
                        print("Terminated after {} iteration(s).".format(self.iteration_count))
                    break
        
        self.n_epochs=self.iteration_count
        self.model_fitted=True
            
    def get_keras(self):
        """
        =========
        get_keras
        =========
        
        Definition: get_keras(self)
        
        Type: Method of DCON class
        
        This function returns a keras model with the neural network parameters obtained from the DCON algorithm. Run .fit() first.
        """
        if self.model_fitted is not True:
            raise RuntimeError('Model was not fitted')
        model = keras.Sequential([
                keras.layers.Dense(self.n_hidden, activation='relu', input_dim=self.n_inputs),
                keras.layers.Dense(1, use_bias=False)
                ])
        
        model.compile(loss='mse', optimizer='adam')
        
        model.layers[0].set_weights([self.W, self.B])
        model.layers[1].set_weights([self.alpha.reshape(self.n_hidden, 1)])
        
        return model
    
    def evaluate(self, X, y):
        """
        ========
        evaluate
        ========
        
        Definition: evaluate(self, X, y)
        
        Type: Method of DCON class
        
        This function returns the evaluation of the neural network on data (X, y) using the keras function 'evaluate' internally.
        
        Parameters
        ----------
        X : np.array
            Feature matrix
            
        y : np.array
            Corresponding labels
        """
        if self.model_fitted is not True:
            raise RuntimeError('Model was not fitted')
        self._check_data(X, n_dims=2, second_dim=self.n_inputs, name='X')
        y = np.squeeze(y)
        self._check_data(y, n_dims=1, name='y')
        if len(X) != len(y):
            raise ValueError('X and y do not have the same number of samples')
        if self.keras_model is None:
            self.keras_model = self.get_keras()

        return self.keras_model.evaluate(X, y, verbose=0)
    
    def predict(self, X):
        """
        =======
        predict
        =======
        
        Definition: predict(self, X)
        
        Type: Method of DCON class
        
        This function returns the predictions of the neural network on data X using the keras function 'predict' internally.
        
        Parameters
        ----------
        X : np.array
            Feature matrix
        """
        if self.model_fitted is not True:
            raise RuntimeError('Model was not fitted')
        self._check_data(X, n_dims=2, second_dim=self.n_inputs, name='X')
        if self.keras_model is None:
            self.keras_model = self.get_keras()
        
        return self.keras_model.predict(X, verbose=0)
            
            