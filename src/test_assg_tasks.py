import numpy as np
import pandas as pd
import sklearn
#import unittest
from twisted.trial import unittest
from assg_tasks import hello_world
from assg_tasks import basic_sigmoid
from assg_tasks import sigmoid
from assg_tasks import sigmoid_grad
from assg_tasks import image2vector
from assg_tasks import normalize_rows
from assg_tasks import softmax
from assg_tasks import l2_loss
from assg_tasks import l1_loss


class test_hello_world(unittest.TestCase):

    def setUp(self):
        pass

    def test_john(self):
        s = hello_world("John")
        self.assertEqual(s, "Hello John!")
        self.assertIsInstance(s, str)

    def test_paul(self):
        s = hello_world("Paul")
        self.assertEqual(s, "Hello Paul!")
        self.assertIsInstance(s, str)

    def test_george(self):
        s = hello_world("George")
        self.assertEqual(s, "Hello George!")
        self.assertIsInstance(s, str)

    def test_ringo(self):
        s = hello_world("Ringo")
        self.assertEqual(s, "Hello Ringo!")
        self.assertIsInstance(s, str)

class test_basic_sigmoid(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_3(self):
        s = basic_sigmoid(3)
        self.assertAlmostEqual(s, 0.9525741268224334)

    def test_input_0(self):
        s = basic_sigmoid(0)
        self.assertAlmostEqual(s, 0.5)
        
    def test_input_neg5(self):
        s = basic_sigmoid(-5)
        self.assertAlmostEqual(s, 0.0066928509242848554)

    def test_using_math_library(self):
        # if using math library should not be vectorized, expect an exception
        # if we pass a numpy array
        x = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            basic_sigmoid(x)

class test_sigmoid(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_scalar(self):
        s = sigmoid(3)
        self.assertAlmostEqual(s, 0.9525741268224334)

    def test_input_vector(self):
        # test a 1-d vector
        x = np.array([-5, 0, 3])
        s = sigmoid(x)
        self.assertTrue(np.allclose(s, np.array([0.00669285, 0.5, 0.95257413])))

    def test_input_matrix(self):
        # test a 3-d tensor
        x = np.linspace(-5, 5, 27).reshape((3,3,3))
        s = sigmoid(x)
        expected_s = np.array(
            [[[0.00669285, 0.00980136, 0.01433278],
              [0.02091496, 0.03042661, 0.04406926],
              [0.06342879, 0.09048789, 0.12751884]],
            
             [[0.17675903, 0.23978727, 0.31664553],
              [0.40501421, 0.5,        0.59498579],
              [0.68335447, 0.76021273, 0.82324097]],
            
             [[0.87248116, 0.90951211, 0.93657121],
              [0.95593074, 0.96957339, 0.97908504],
              [0.98566722, 0.99019864, 0.99330715]]]
        )
        self.assertTrue(np.allclose(s, expected_s))

    def test_input_list(self):
        # a regular list still does not work for a vectorized function
        x = [-5, 0, 3]
        with self.assertRaises(TypeError):
            s = sigmoid(x)
        

class test_sigmoid_grad(unittest.TestCase):

    def setUp(self):
        pass

    def test_input_scalar(self):
        s = sigmoid_grad(3)
        self.assertAlmostEqual(s, 0.045176659730912)

    def test_input_vector(self):
        # test a 1-d vector
        x = np.array([-5, 0, 3])
        s = sigmoid_grad(x)
        self.assertTrue(np.allclose(s, np.array([0.00664806, 0.25, 0.04517666])))

    def test_input_matrix(self):
        # test a 3-d tensor
        x = np.linspace(-5, 5, 27).reshape((3,3,3))
        s = sigmoid_grad(x)
        expected_s = np.array(
        [[[0.00664806, 0.00970529, 0.01412736],
          [0.02047752, 0.02950084, 0.04212716],
          [0.05940558, 0.08229983, 0.11125779]],
        
         [[0.14551528, 0.18228933, 0.21638114],
          [0.2409777,  0.25,       0.2409777 ],
          [0.21638114, 0.18228933, 0.14551528]],
        
         [[0.11125779, 0.08229983, 0.05940558],
          [0.04212716, 0.02950084, 0.02047752],
          [0.01412736, 0.00970529, 0.00664806]]]
        )
        self.assertTrue(np.allclose(s, expected_s))

    def test_input_list(self):
        # a regular list still does not work for a vectorized function
        x = [-5, 0, 3]
        with self.assertRaises(TypeError):
            s = sigmoid_grad(x)

class test_image2vector(unittest.TestCase):

    def setUp(self):
        pass

    def test_matrix(self):
        image = np.array(
            [[[ 0.67826139,  0.29380381],
                [ 0.90714982,  0.52835647],
                [ 0.4215251 ,  0.45017551]],
        
               [[ 0.92814219,  0.96677647],
                [ 0.85304703,  0.52351845],
                [ 0.19981397,  0.27417313]],
        
               [[ 0.60659855,  0.00533165],
                [ 0.10820313,  0.49978937],
                [ 0.34144279,  0.94630077]]]
        )
        v = image2vector(image)
        expected_v = np.array(
            [[0.67826139],
             [0.29380381],
             [0.90714982],
             [0.52835647],
             [0.4215251 ],
             [0.45017551],
             [0.92814219],
             [0.96677647],
             [0.85304703],
             [0.52351845],
             [0.19981397],
             [0.27417313],
             [0.60659855],
             [0.00533165],
             [0.10820313],
             [0.49978937],
             [0.34144279],
             [0.94630077]]            
        )
        self.assertTrue(np.allclose(v, expected_v))
        self.assertTrue(v.shape == (18, 1))
        self.assertTrue(v.ndim == 2)

    def test_random_matrix(self):
        image = np.random.random((256, 128, 3))
        v = image2vector(image)
        self.assertTrue(v.shape == (98304, 1))
        self.assertAlmostEqual(v[0,0], image[0,0,0])
        self.assertAlmostEqual(v[98303,0], image[255, 127, 2])

class test_normalize_rows(unittest.TestCase):

    def setUp(self):
        pass

    def test_matrix(self):
        x = np.array(
            [[0, 3, 4],
             [1, 6, 4]]
        )
        expected_x_norm = np.array(
            [[0.,         0.6,        0.8       ],
             [0.13736056, 0.82416338, 0.54944226]]            
        )
        x_norm = normalize_rows(x)
        self.assertTrue(np.allclose(x_norm, expected_x_norm))
        self.assertTrue(x_norm.shape == (2,3))
        # the rows are unit vectors, so sum of squares should all be 1
        self.assertTrue(np.allclose(np.sum(x_norm**2.0, axis=1), np.array([1.0, 1.0]))) 
        # check that original x was not modified
        self.assertFalse(np.allclose(x_norm, x))
        

    def test_random_matrix(self):
        x = np.random.random((20,10))
        x_norm = normalize_rows(x)
        self.assertTrue(x_norm.shape == (20,10))
        # the rows are unit vectors, so sum of squares should all be 1
        self.assertTrue(np.allclose(np.sum(x_norm**2.0, axis=1), np.ones((20,))))
        # check that original x was not modified
        self.assertFalse(np.allclose(x_norm, x))
                                    
class test_softmax(unittest.TestCase):

    def setUp(self):
        pass

    def test_matrix(self):
        x = np.array(
            [[9, 2, 5, 0, 0],
             [7, 5, 0, 0 ,0]]
        )
        expected_s = np.array(
            [[9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04],
             [8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]]
        )
        s = softmax(x)
        self.assertTrue(np.allclose(s, expected_s))
        self.assertTrue(s.shape == (2,5))
        # the rows of softmax are a normalized probability space so should sum up to 0
        self.assertTrue(np.allclose(np.sum(s, axis=1), np.array([1.0, 1.0]))) 
        # check that original x was not modified
        self.assertFalse(np.allclose(s, x))
        
    def test_random_matrix(self):
        x = np.random.random((20,10))
        s = softmax(x)
        self.assertTrue(s.shape == (20,10))
        # the rows of softmax are a normalized probability space so should sum up to 0
        self.assertTrue(np.allclose(np.sum(s, axis=1), np.ones((20,)))) 
        # check that original x was not modified
        self.assertFalse(np.allclose(s, x))

class test_l2_loss(unittest.TestCase):

    def setUp(self):
        pass

    def test_case1(self):
        y_pred = np.array([.9, 0.2, 0.1, .4, .9])
        y_true = np.array([1, 0, 0, 1, 1])
        loss = l2_loss(y_pred, y_true)
        self.assertAlmostEqual(loss, 0.43)

    def test_case2(self):
        y_pred = np.array([0.8, 0.3, 0.7, 0.5, 0.8, 0.6, 0.5, 0.2])
        y_true = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        loss = l2_loss(y_pred, y_true)
        self.assertAlmostEqual(loss, 1.16)

class test_l1_loss(unittest.TestCase):

    def setUp(self):
        pass

    def test_case1(self):
        y_pred = np.array([.9, 0.2, 0.1, .4, .9])
        y_true = np.array([1, 0, 0, 1, 1])
        loss = l1_loss(y_pred, y_true)
        self.assertAlmostEqual(loss, 1.1)

    def test_case2(self):
        y_pred = np.array([0.8, 0.3, 0.7, 0.5, 0.8, 0.6, 0.5, 0.2])
        y_true = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        loss = l1_loss(y_pred, y_true)
        self.assertAlmostEqual(loss, 2.8)

