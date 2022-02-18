namespace Common.Class
{
    public class Student
    {
        private string pIdentifier;
        private string pLogin;
        private string pPassword;
        private string pUserDefaultLanguage;
        private string pFirstName;
        private string pLastName;
        private string pUserMail;
        private string pUserRoles;
        private string pUserUILanguage;

        public string Identifier { set { pIdentifier = value; } get { return pIdentifier; } }
        public string Login { set { pLogin = value; } get { return pLogin; } }
        public string Password { set { pPassword = value; } get { return pPassword; } }
        public string UserDefaultLanguage { set { pUserDefaultLanguage = value; } get { return pUserDefaultLanguage; } }
        public string FirstName { set { pFirstName = value; } get { return pFirstName; } }
        public string LastName { set { pLastName = value; } get { return pLastName; } }
        public string UserMail { set { pUserMail = value; } get { return pUserMail; } }
        public string UserRoles { set { pUserRoles = value; } get { return pUserRoles; } }
        public string UserUILanguage { set { pUserUILanguage = value; } get { return pUserUILanguage; } }

        public Student() { }

        public override string ToString()
        {
            return
                "TestTaker: {" +
                    "Identifier: " + Identifier + ", " +
                    "Login: " + Login + ", " +
                    "Password: " + Password + ", " +
                    "UserDefaultLanguage: " + UserDefaultLanguage + ", " +
                    "FirstName: " + FirstName + ", " +
                    "LastName: " + LastName + ", " +
                    "UserMail: " + UserMail + ", " +
                    "UserRoles: " + UserRoles + ", " +
                    "UserUILanguage: " + UserUILanguage +
                "}";
        }
    }
}
