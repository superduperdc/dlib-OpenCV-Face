imshow (insertShape(frame,'rectangle',bbox));
imshow (insertMarker(frame,frame1,'+'));
imshow (insertMarker(frame,[mean(frame1(1:2,:)); mean(frame1(3:4,:))],'o'));