Author:201:10 Title:201:22 Venue:201:42 SameAuthor:10:10 SameTitle:22:22 SameVenue:42:42 SameBib:201:201 
4
11:SameTitle-id1,Title-id1,SameTitle-id0
100:Title-id0,Venue-id0,Author-id0,SameBib-id1,SameBib-id0
5:SameAuthor-id0,SameAuthor-id1,Author-id1
21:SameVenue-id0,Venue-id1,SameVenue-id1
Author:1,0:2,0 Title:1,0:0,0 Venue:1,0:3,0 SameAuthor:2,0:2,0 SameTitle:0,0:0,0 SameVenue:3,0:3,0 SameBib:1,0:1,0 
14
0:!Author(bc1,a1) v !Author(bc2,a2) v !SameAuthor(a1,a2) v SameBib(bc1,bc2) 
0:!Title(bc1,t1) v !Title(bc2,t2) v !SameTitle(t1,t2) v SameBib(bc1,bc2) 
0:!Venue(bc1,t1) v !Venue(bc2,t2) v !SameVenue(t1,t2) v SameBib(bc1,bc2) 
0:!Author(bc1,a1) v !Author(bc2,a2) v !SameBib(bc1,bc2) v SameAuthor(a1,a2) 
0:!Title(bc1,t1) v !Title(bc2,t2) v !SameBib(bc1,bc2) v SameTitle(t1,t2) 
0:!Venue(bc1,t1) v !Venue(bc2,t2) v !SameBib(bc1,bc2) v SameVenue(t1,t2)
0:!SameBib(b1,b2) v !SameBib(b2,b3) v SameBib(b1,b3)
0:!SameAuthor(a1,a2) v !SameAuthor(a2,a3) v SameAuthor(a1,a3)
0:!SameTitle(t1,t2) v !SameTitle(t2,t3) v SameTitle(t1,t3)
0:!SameVenue(t1,t2) v !SameVenue(t2,t3) v SameVenue(t1,t3)
0:!SameBib(b1,b2)
0:!SameAuthor(a1,a2)
0:!SameTitle(t1,t2)
0:!SameVenue(t1,t2)
