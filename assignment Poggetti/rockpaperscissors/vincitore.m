function r = vincitore(a,b)
    % 1 Sasso
    % 2 Carta
    % 3 Forbici
    % 4 Lizard
    % 5 Spock
    

    switch a
            case 1 
                if b == 2 || b == 5
                     r = -1;
                
                elseif b == 4 || b == 3
                         r = 1;
                
                
                else 
                       r = 0;

                end

            case 2 
                if b == 4 || b == 3
                     r = -1;
                
                elseif b == 1 || b == 5
                         r = 1;
             
                else 
                       r = 0;

                end

            case 3 
                if b == 1 || b == 5
                     r = -1;
                
                elseif b == 2 || b == 4
                         r = 1;
             
                else 
                       r = 0;

                end

            case 4 
                if b == 1 || b == 3
                     r = -1;
                
                elseif b == 2 || b == 5
                         r = 1;
             
                else 
                       r = 0;

                end

            case 5
                if b == 4 || b == 2
                     r = -1;
                
                elseif b == 3 || b == 1
                         r = 1;
             
                else 
                       r = 0;

                end
    end
              

