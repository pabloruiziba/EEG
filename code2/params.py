def getParams():

	params ={}

	#download Data
	params["dataEDF"] = "data/EDF/"
	params["webPage"] = "http://www.physionet.org/pn4/eegmmidb/"
	
	#visualize multiple signal
	params["plotsMS"] = "plots/multipleSignal/"
	params["pacient"] = [10] #1-109
 	params["experiment"] = [1,2] #1-14


	params["Filter"] ={"upFilter":50,"downFilter":0} 


	return params

def getCoords():
	return {"1":[96.875,-178.125],
			"2":[115.625,-181.25],
			"3":[131.25,-181.25],
			"43":[56.25,-200.0],
			"41":[75.0,-200.0],
			"8":[93.75,-200.0],
			"9":[112.5,-200.0],
			"10":[131.25,-200.0],
			"11":[150.0,-200.0],
			"12":[168.75,-200.0],
			"13":[187.5,-200.0],
			"14":[206.25,-200.0],
			"42":[225.0,-200.0],
			"44":[243.75,-200.0],
			"18":[150.0,-218.75],
			"51":[150.0,-237.5],
			"58":[150.0,-256.25],
			"4":[150.0,-181.25],
			"34":[150.0,-162.5],
			"27":[150.0,-143.75],
			"39":[78.125,-175.0],
			"5":[168.75,-181.25],
			"6":[184.375,-181.25],
			"7":[203.125,-178.125],
			"40":[221.875,-175.0],
			"33":[134.375,-159.375],
			"32":[118.75,-162.5],
			"31":[103.125,-159.375],
			"30":[87.5,-156.25],
			"35":[165.625,-159.375],
			"36":[181.25,-162.5],
			"37":[196.875,-159.375],
			"38":[212.5,-156.25],
			"26":[118.75,-143.75],
			"28":[178.125,-143.75],
			"25":[100.0,-137.5],
			"29":[196.875,-137.5],
			"22":[128.125,-128.125],
			"24":[171.875,-128.125],
			"17":[131.25,-221.875],
			"16":[112.5,-225.0],
			"19":[168.75,-221.875],
			"20":[187.5,-225.0],
			"15":[93.75,-225.0],
			"21":[206.25,-225.0],
			"46":[225.0,-228.125],
			"45":[75.0,-228.125],
			"50":[134.375,-240.625],
			"49":[115.625,-240.625],
			"48":[96.875,-243.75],
			"47":[78.125,-246.875],
			"52":[165.625,-240.625],
			"53":[184.375,-240.625],
			"54":[203.125,-243.75],
			"55":[221.875,-246.875],
			"57":[121.875,-259.375],
			"56":[93.75,-262.5],
			"59":[178.125,-259.375],
			"60":[206.25,-262.5],
			"62":[150.0,-275.0],
			"23":[150.0,-121.875],
			"64":[150.0,-293.75],
			"63":[171.875,-278.125],
			"61":[128.125,-278.125]}


#all labels with the exacly name
#"Fc5.","Fc3.","Fc1.","Fcz.","Fc2.","Fc4.","Fc6.",
#"C5..","C3..","C1..","Cz..","C2..","C4..","C6..",
#"Cp5.","Cp3.","Cp1.","Cpz.","Cp2.","Cp4.","Cp6.",
#"Fp1.","Fpz.","Fp2.",
#"Af7.","Af3.","Afz.","Af4.","Af8.",
#"F7..","F5..","F3..","F1..","Fz..","F2..","F4..","F6..","F8..",
#"Ft7.","Ft8.",
#"T7..","T8..","T9..","T10.",
#"Tp7.","Tp8.",
#"P7..","P5..","P3..","P1..",
#"Pz..",
#"P2..","P4..","P6..","P8..",
#"Po7.","Po3.","Poz.","Po4.","Po8.",
#"O1..","Oz..","O2..",
#"Iz.."
