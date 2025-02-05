import requests
import json

def test_classification(fen_string):
    url = "http://localhost:8000/classify"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "fen": fen_string
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises an exception for 4XX/5XX status codes
        
        # Pretty print the results
        result = response.json()
        print("\nFEN:", fen_string)
        print("Opening:", result['opening'])
        print("Certainty: {:.1f}%".format(result['certainty']))
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError testing FEN: {fen_string}")
        print(f"Error details: {str(e)}")
        if response := getattr(e, 'response', None):
            print(f"Response: {response.text}")
        return False

def main():
    # List of test FEN strings (from common openings)
    test_fens = [
        # Ruy Lopez (after 3...a6)
        "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        
        # Sicilian Dragon (after 4.Nf3)
        "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",
        
        # King's Indian Defense (after 4.Nf3)
        "rnbqk2r/ppp1ppbp/3p1np1/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 2 4",
        
        # Italian Game (after 4.c3)
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 4",
        
        # Caro-Kann Defense (after 4.Nf3)
        "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 1 4",
        
        # French Defense (after 4.e5)
        "rnbqkb1r/ppp2ppp/4pn2/3pP3/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 4"
    ]
    
    print("Starting API tests...")
    successes = 0
    
    for fen in test_fens:
        if test_classification(fen):
            successes += 1
            
    print(f"\nTest Summary:")
    print(f"Successful tests: {successes}/{len(test_fens)}")

if __name__ == "__main__":
    main()