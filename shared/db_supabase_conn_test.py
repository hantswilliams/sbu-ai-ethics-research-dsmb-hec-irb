import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables from .env file
load_dotenv()

def test_supabase_connection():
    try:
        # Connection parameters
        conn_params = {
            'dbname': os.getenv('SUPABASE_DB_NAME', 'postgres'),
            'user': os.getenv('SUPABASE_USER', 'postgres.eigtivpuaudjkdadajfb'),
            'password': os.getenv('SUPABASE_PASSWORD'),
            'host': os.getenv('SUPABASE_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            'port': os.getenv('SUPABASE_PORT', '6543')
        }
        
        # Attempt to establish connection
        print("Attempting to connect to Supabase...")
        conn = psycopg2.connect(**conn_params)
        
        # Create a cursor
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Simple test query
        print("Executing test query...")
        cur.execute('SELECT current_timestamp;')
        result = cur.fetchone()
        print(f"Current database timestamp: {result['current_timestamp']}")
        
        # Close cursor and connection
        cur.close()
        conn.close()
        print("Connection test successful! Connection closed.")
        return True
        
    except Exception as e:
        print(f"Error connecting to Supabase: {str(e)}")
        return False

if __name__ == "__main__":
    test_supabase_connection()
