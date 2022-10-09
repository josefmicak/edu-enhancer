using DomainModel;

namespace DataLayer
{
    public class DataFunctions
    {
        //class that will be used by functions in controller/business layer to work with database in the future
        private readonly CourseContext _context;

        public DataFunctions(CourseContext context)
        {
            _context = context;
        }
    }
}
