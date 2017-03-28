# with open("parsedoutput.txt") as f:
# 	lines = f.readlines()
# 	print lines

lines = [line.rstrip('\n') for line in open('parsedoutput.txt')]
f = open("axispush.txt","a")
for i in range(len(lines)):
	f.write("axis.push_back(Point3f(" + lines[i]+"));\n")
	i=i+1