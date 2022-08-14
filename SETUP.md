You can follow the following steps to setup the project locally.
<ol>
    <li> 
        Fork the Repo. 
    </li>
    <li> 
        Clone the repo locally using <code>git clone https://github.com/{your-username}/machine-learning-platform.git</code> 
    </li>
    <li> 
        Go into the directory using <code>cd machine-learning-platform</code> 
    </li>
    <li> 
        Create a new virtual enviornment <code>python3 -m venv env</code>. If you don't have virtual enviornment install. Install using <code>pip install virtualenv</code>
    </li>
    <li>
        Install the dependencies using <code>pip3 install -r requirments.txt</code>
    </li>
    <li>
        Ready your database for migration <code>python3 manage.py makemigrations</code>
    </li>
    <li>
        migrate <code>python3 manage.py migrate --run-syncdb</code>
    </li>
    <li>
        Start your backend development server <code>python3 manage.py runserver</code>
    </li>
</ol>