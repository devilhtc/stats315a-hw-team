import numpy
import util

def main():
	Util = util.Util
	u = Util()
	testUtil(u)

def testUtil(u):
	#print('hello world!')
	c_0, c_1 = u.gen_cent()
	data = u.gen_data(c_0, 100)
	print (data)

if __name__=='__main__':
	main()
